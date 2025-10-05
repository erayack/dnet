"""Layer weight management with memory-mapped files and prefetching."""

import ctypes
import ctypes.util
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Set

import mlx.core as mx

from .logger import logger
from .model import (
    MappedFile,
    ModelMetadata,
    get_model_layer_name,
    load_weight,
)

# Load libc for madvise
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int

# macOS madvise constants
MADV_SEQUENTIAL = 2  # Sequential access
MADV_WILLNEED = 3  # Prefetch pages
MADV_DONTNEED = 4  # Pages not needed


class LayerManager:
    """Manages memory-mapped LLM layers with prefetching.

    This is built around the assumption that API servers handle the embedding
    and output projection. Since embedding tokens and LM heads are generally
    ~1GB even for big models like Llama-65B, they fit in RAM without memory
    mapping. This class is meant for shard servers which hold the layers.
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        assigned_layers: List[int],
        thread_pool_size: int = 2,
    ) -> None:
        """Initialize layer manager.

        Args:
            model_metadata: Model metadata containing weight info
            assigned_layers: Layer indices assigned to this manager
            thread_pool_size: Number of threads for prefetching
        """
        # Extract the name of safetensors associated with the layers
        self.assigned_layers: Set[int] = set(assigned_layers)

        # Open memory-mapped files
        self.weight_info = weight_info = model_metadata.weight_info
        filenames: Set[str] = set(
            wts.filename
            for layer in assigned_layers
            for wts in weight_info[layer].values()
        )
        self.mapped_files: Dict[str, MappedFile] = {
            fname: MappedFile(fname) for fname in filenames
        }

        # Prefetch executor
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)

        logger.info(f"Initialized layer manager with layers {self.assigned_layers}")

    def _memadvise_layer(self, layer_idx: int, memadvise: int) -> bool:
        """Apply madvise to all weights in a layer.

        Args:
            layer_idx: Layer index
            memadvise: madvise constant (MADV_*)

        Returns:
            True if all madvise calls succeeded
        """
        if layer_idx not in self.assigned_layers:
            return False

        # Get information about tensors that this layer needs
        weight_data = self.weight_info[layer_idx]

        # Loop over each tensor and apply madvise
        success = True
        for wt in weight_data.values():
            offset, size = wt.offset, wt.size_bytes
            mapped_file = self.mapped_files[wt.filename]
            layer_addr = mapped_file.base_addr + offset

            # Use madvise to prefetch
            result = libc.madvise(layer_addr, size, memadvise)
            success &= result == 0
        return success

    def prefetch_layer(self, layer_idx: int) -> bool:
        """Prefetch a layer using madvise.

        Args:
            layer_idx: Layer index to prefetch

        Returns:
            True if prefetch succeeded
        """
        result = self._memadvise_layer(layer_idx, MADV_WILLNEED)
        if result:
            logger.info(f"Prefetched layer {layer_idx}")
        else:
            logger.info(f"Failed to prefetch layer {layer_idx}")
        return result

    def release_layer(self, layer_idx: int) -> bool:
        """Mark layer as not needed anymore.

        Args:
            layer_idx: Layer index to release

        Returns:
            True if release succeeded
        """
        result = self._memadvise_layer(layer_idx, MADV_DONTNEED)
        if result:
            logger.info(f"Released layer {layer_idx}")
        else:
            logger.info(f"Failed to release layer {layer_idx}")
        return result

    def load_layer_to_gpu(self, layer_idx: int) -> Dict[str, mx.array]:
        """Load layer from memory map to GPU.

        Args:
            layer_idx: Layer index to load

        Returns:
            Dictionary of weight name to MLX array

        Raises:
            RuntimeError: If layer is not assigned to this manager
        """
        if layer_idx not in self.assigned_layers:
            raise RuntimeError(f"layer {layer_idx} not assigned to this node")

        # Get information about tensors that this layer needs
        weight_data = self.weight_info[layer_idx]

        # Load each tensor
        data: Dict[str, mx.array] = {}
        for name, wt in weight_data.items():
            data[get_model_layer_name(layer_idx, name)] = load_weight(
                wt, self.mapped_files
            )
        return data

    def async_prefetch(self, layer_idx: int) -> Future:
        """Asynchronously prefetch a layer.

        Args:
            layer_idx: Layer index to prefetch

        Returns:
            Future that completes when prefetch is done
        """
        return self.executor.submit(self.prefetch_layer, layer_idx)

    def close(self) -> None:
        """Clean up resources."""
        self.executor.shutdown(wait=True)

        for mapped_file in self.mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()
