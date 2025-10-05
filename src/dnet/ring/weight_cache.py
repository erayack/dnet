"""Layer weight cache with windowed GPU residency and LRU eviction."""

import threading
import time
from typing import Dict, List, Optional

import mlx.core as mx

from ..utils.layer_manager import LayerManager
from ..utils.logger import logger
from ..utils.model import ModelMetadata


class WeightCache:
    """Layer weight cache with windowed GPU residency and LRU eviction.

    Only up to `window_size` layers are kept resident in device memory. All
    other layers remain memory-mapped on disk and can be preheated via
    `madvise` using the layer manager to reduce page faults on load.
    """

    def __init__(
        self,
        assigned_layers: List[int],
        model_metadata: ModelMetadata,
        window_size: Optional[int] = None,
    ) -> None:
        """Initialize weight cache.

        Args:
            assigned_layers: Layer indices assigned to this cache
            model_metadata: Model metadata containing weight info
            window_size: Maximum number of layers to keep resident (None = all)
        """
        self.assigned_layers = assigned_layers
        # Cap the resident cache to the window size if provided
        self.max_weights = (
            int(window_size)
            if (window_size is not None and window_size > 0)
            else len(self.assigned_layers)
        )
        self.cache: Dict[int, tuple[Dict[str, mx.array], float]] = (
            {}
        )  # layer_id -> (data, access_time)
        self.reference_counts: Dict[int, int] = {}  # layer_id -> count
        self.layer_manager = LayerManager(model_metadata, assigned_layers)
        self.lock = threading.Lock()

    def get_weight(self, layer_id: int) -> Optional[Dict[str, mx.array]]:
        """Get weight from cache.

        Args:
            layer_id: Layer index

        Returns:
            Dictionary of weight name to MLX array, or None if load failed
        """
        with self.lock:
            if layer_id in self.cache:
                # Update access time
                data, _ = self.cache[layer_id]
                self.cache[layer_id] = (data, time.time())
                self.reference_counts[layer_id] = (
                    self.reference_counts.get(layer_id, 0) + 1
                )
                logger.debug(
                    f"Cache hit for layer {layer_id}, "
                    f"ref count: {self.reference_counts[layer_id]}"
                )
                return data

            # Need to load new weight
            if len(self.cache) >= self.max_weights:
                self._evict_lru()

            try:
                data = self.layer_manager.load_layer_to_gpu(layer_id)
                self.cache[layer_id] = (data, time.time())
                self.reference_counts[layer_id] = 1
                logger.info(f"Loaded weights for layer {layer_id} into cache")
                return data
            except Exception as e:
                logger.exception(f"Failed to load weight {layer_id}: {e}")
                return None

    def decrease_reference(self, layer_id: int) -> None:
        """Decrease reference count for layer.

        Args:
            layer_id: Layer index
        """
        with self.lock:
            if layer_id in self.reference_counts:
                self.reference_counts[layer_id] -= 1
                logger.debug(
                    f"Decreased ref count for {layer_id}: "
                    f"{self.reference_counts[layer_id]}"
                )

    def prefetch_to_ram(self, layer_id: int) -> Optional[object]:
        """Asynchronously hint the OS to prefetch layer pages into RAM.

        This does not allocate device memory; it only warms the file-backed
        pages to speed up subsequent `get_weight` loads.

        Args:
            layer_id: Layer index to prefetch

        Returns:
            Future object, or None if prefetch failed
        """
        try:
            return self.layer_manager.async_prefetch(layer_id)
        except Exception as e:
            logger.debug(f"Prefetch to RAM failed for layer {layer_id}: {e}")
            return None

    def _evict_lru(self) -> None:
        """Evict least recently used weight with zero references."""
        candidates = [
            (lid, access_time)
            for lid, (_, access_time) in self.cache.items()
            if self.reference_counts.get(lid, 0) == 0
        ]

        if candidates:
            # Sort by access time, evict oldest
            candidates.sort(key=lambda x: x[1])
            layer_id = candidates[0][0]

            # Hint OS we no longer need these pages in RAM
            try:
                self.layer_manager.release_layer(layer_id)
            except Exception:
                pass

            # Remove from cache
            del self.cache[layer_id]
            if layer_id in self.reference_counts:
                del self.reference_counts[layer_id]

            logger.info(f"Evicted layer {layer_id} from cache")

    def evict_layer(self, layer_id: int) -> bool:
        """Proactively evict a specific layer if it has no active references.

        Args:
            layer_id: Layer index to evict

        Returns:
            True if evicted, False if layer has active references
        """
        with self.lock:
            if self.reference_counts.get(layer_id, 0) != 0:
                return False
            if layer_id not in self.cache:
                return True
            try:
                self.layer_manager.release_layer(layer_id)
            except Exception:
                pass
            del self.cache[layer_id]
            if layer_id in self.reference_counts:
                del self.reference_counts[layer_id]
            logger.info(f"Proactively evicted layer {layer_id} from cache")
            return True
