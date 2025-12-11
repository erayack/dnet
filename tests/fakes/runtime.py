"""Runtime-focused fakes used by shard/ring/policy tests."""

from __future__ import annotations

from typing import Any


class FakeRuntimeMinimal:
    """Tiny runtime with input/output pools and wire dtype defaults."""

    def __init__(self, node_id: str = "node-1"):
        self.shard_id: str = node_id
        self.assigned_layers: list[int] = []
        self.model: Any = None
        self.model_path: str | None = None
        self._queue: list = []
        self._cache_reset: bool = False
        import mlx.core as mx
        from dnet.core.memory.memory_pool import LayerAwareMemoryPool

        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self.input_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.output_pool = LayerAwareMemoryPool(total_memory_mb=2)

    def attach_loop(self, loop):
        self._loop = loop

    def start(self):
        self._started = True

    def shutdown(self):
        self._started = False

    def reset_cache(self):
        self._cache_reset = True

    def queue_size(self) -> int:
        return len(self._queue)

    def load_model_core(self, req):
        self.assigned_layers = list(req.layers)
        self.model_path = req.model_path
        self.model = object()

    def unload_model_core(self):
        from dnet.shard.models import ShardUnloadModelResponse

        self.model = None
        return ShardUnloadModelResponse(success=True, message="ok")


class FakeRuntimeForAdapter:
    """Runtime stub for RingAdapter tests (queues + executor)."""

    def __init__(
        self, shard_id: str = "S1", max_queue_size: int = 8, assigned_next=set()
    ):
        import mlx.core as mx
        import queue as pyq
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from dnet.core.memory.memory_pool import LayerAwareMemoryPool
        from dnet.shard.config import ComputeConfig

        self.shard_id = shard_id
        self.max_queue_size = max_queue_size
        # Use more than one worker so ingest/compute don't starve each other
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.activation_recv_queue = pyq.Queue(maxsize=max_queue_size)
        self.activation_send_queue = pyq.Queue(maxsize=max_queue_size)
        self._assigned_set = set(assigned_next)
        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self._kv_calls: list[str] = []

        # Attributes needed for OffloadPolicy
        self.input_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.output_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.model = None
        self.model_metadata = None
        self.model_path: str | None = None
        self._model_lock = threading.Lock()
        self._mlx_lock = threading.Lock()
        self.prefetch_threads = 1
        self.compute_config = ComputeConfig()
        self.assigned_layers: list[int] = []
        self._assigned_sorted: list[int] = []
        self.policy = None
        self._compute_busy = threading.Event()
        self._loop = None

        # Compute worker state
        self._compute_running = False
        self._compute_task: Any = None

    def get_or_make_kv(self, nonce: str):
        self._kv_calls.append(nonce)
        return []

    async def start_compute_worker(self):
        """Start async worker that processes activation_recv_queue via policy."""
        import asyncio
        import queue as pyq

        self._compute_running = True
        self._loop = asyncio.get_running_loop()

        async def _worker():
            loop = asyncio.get_running_loop()
            while self._compute_running:
                try:
                    msg = await loop.run_in_executor(
                        self.executor,
                        lambda: self.activation_recv_queue.get(timeout=0.1),
                    )
                    if self.policy:
                        await loop.run_in_executor(
                            self.executor, self.policy.process, msg
                        )
                except pyq.Empty:
                    await asyncio.sleep(0.01)
                except Exception:
                    await asyncio.sleep(0.01)

        self._compute_task = asyncio.create_task(_worker())

    async def stop_compute_worker(self):
        """Stop the compute worker."""
        import asyncio

        self._compute_running = False
        if self._compute_task:
            try:
                await asyncio.wait_for(self._compute_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._compute_task.cancel()
                try:
                    await self._compute_task
                except asyncio.CancelledError:
                    pass
        self._compute_task = None

    def close(self):
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def emit_result(self, msg):
        """Mirror ShardRuntime.emit_result for adapter tests."""
        try:
            self.activation_send_queue.put_nowait(msg)
        except Exception:
            self.activation_send_queue.put(msg)


class FakeRuntimeForPolicy:
    """Runtime stub for policy tests with pools, locks, and emit_result capture."""

    def __init__(self, assigned_layers=None, num_layers: int = 4, shard_id: str = "S1"):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import mlx.core as mx
        from dnet.core.memory.memory_pool import LayerAwareMemoryPool
        from dnet.shard.config import ComputeConfig
        from .policies import FakeComputeModel
        import types

        self.shard_id = shard_id
        self.assigned_layers = list(assigned_layers or [1, 2])
        self._assigned_sorted = sorted(self.assigned_layers)
        self._assigned_set = set(self._assigned_sorted)
        self.model_metadata = types.SimpleNamespace(num_layers=int(num_layers))
        self.model = FakeComputeModel(mx)
        self._mlx_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self.prefetch_threads = 1
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.compute_config = ComputeConfig()
        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self.input_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.output_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self._emitted: list = []
        self._compute_busy = threading.Event()
        self._loop = None

    def attach_loop(self, loop):
        self._loop = loop

    def get_or_make_kv(self, nonce: str):
        return []

    def emit_result(self, msg):
        self._emitted.append(msg)

    def close(self):
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
