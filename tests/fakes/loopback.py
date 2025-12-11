"""Loopback fakes for pipelined ring testing without real networking."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Optional

import queue as pyq


class LoopbackRingStub:
    """Stub that implements StreamActivations by routing frames to a target adapter's ingress_q.

    This allows building multi-adapter pipelines without real gRPC connections.
    """

    def __init__(self, target_adapter: Any, shared_log: Optional[dict] = None):
        """Create a loopback stub targeting the given RingAdapter.

        Args:
            target_adapter: The downstream RingAdapter whose ingress_q will receive frames
            shared_log: Optional shared delivery log dict for test assertions
        """
        self.target_adapter = target_adapter
        # Use shared log if provided, otherwise create own log
        self.delivery_log: dict[str, list[tuple[int, float]]] = (
            shared_log if shared_log is not None else defaultdict(list)
        )

    def StreamActivations(self, request_iterator):
        """Consume ActivationFrames from the client stream and route to target ingress_q."""

        async def _loopback_stream():
            try:
                async for frame in request_iterator:
                    # Extract the ActivationRequest from the frame
                    request = frame.request
                    seq = frame.seq

                    # Track delivery for assertions
                    timestamp = time.time()
                    self.delivery_log[request.nonce].append((seq, timestamp))

                    # Push directly into target adapter's ingress queue
                    await self.target_adapter.admit_frame(request)

                    # Yield ack (though RingAdapter doesn't currently process these)
                    from dnet.protos.dnet_ring_pb2 import StreamAck

                    yield StreamAck(
                        nonce=request.nonce,
                        seq=seq,
                        accepted=True,
                        message="",
                    )
            except Exception as e:
                # Log but don't crash the test
                import logging

                logging.debug(f"Loopback stream error: {e}")

        return _loopback_stream()

    def get_delivery_log(self) -> dict[str, list[tuple[int, float]]]:
        """Return the delivery log for assertions."""
        return dict(self.delivery_log)


class SimulatedStreamCtx:
    """Simulated StreamContext that mirrors the shape from StreamManager.

    This allows tests to control stream state without real gRPC calls.
    """

    def __init__(
        self,
        nonce: str,
        acks: list[tuple[int, bool]] | None = None,
    ):
        """Create a simulated stream context.

        Args:
            nonce: The nonce identifier for this stream
        """
        self.nonce = nonce
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.call: Optional[Any] = None
        self.ack_task: Optional[asyncio.Task] = None
        self.acks: list[tuple[int, bool]] = acks if acks is not None else []
        self.open = True
        self.disabled = False
        self.disabled_until = 0.0
        self.last_seq = 0
        self.last_activity_t = time.time()

    def set_disabled(self, flag: bool):
        """Set the disabled flag to simulate stream closure."""
        self.disabled = flag
        if flag:
            loop = asyncio.get_running_loop()
            self.disabled_until = loop.time() + 1.0


class SlowFakeRuntime:
    """FakeRuntime extension with tunable delays for heterogeneous device simulation.

    Extends FakeRuntimeForAdapter with compute and prefetch delays to simulate
    different device speeds in pipelined tests.
    """

    def __init__(
        self,
        shard_id: str = "S1",
        max_queue_size: int = 8,
        assigned_next: set = None,
        compute_delay_s: float = 0.0,
        prefetch_delay_s: float = 0.0,
    ):
        """Create a slow fake runtime.

        Args:
            shard_id: Shard identifier
            max_queue_size: Maximum queue size
            assigned_next: Set of layer IDs assigned to this shard
            compute_delay_s: Simulated compute delay in seconds
            prefetch_delay_s: Simulated prefetch delay in seconds
        """
        import mlx.core as mx
        from dnet.core.memory.memory_pool import LayerAwareMemoryPool
        from concurrent.futures import ThreadPoolExecutor

        self.shard_id = shard_id
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.activation_recv_queue = pyq.Queue(maxsize=max_queue_size)
        self.activation_send_queue = pyq.Queue(maxsize=max_queue_size)
        self._assigned_set = set(assigned_next or set())
        self._assigned_sorted = sorted(self._assigned_set)
        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self._kv_calls: list[str] = []
        self.input_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.output_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.model = None
        self.model_metadata = None
        self.model_path: str | None = None

        # Timing attributes
        self.compute_delay_s = float(compute_delay_s)
        self.prefetch_delay_s = float(prefetch_delay_s)
        self.timestamps: list[float] = []

        # For async worker simulation
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._worker_started = False

    def get_or_make_kv(self, nonce: str):
        """Track KV initialization calls."""
        if nonce not in self._kv_calls:
            self._kv_calls.append(nonce)
        return []

    async def ensure_worker_started(self):
        """Idempotently start the worker loop."""
        if not self._worker_started:
            await self.start_worker()

    async def start_worker(self):
        """Start a worker that processes recv_queue with delays and enqueues to send_queue."""
        if self._running:
            return
        self._running = True
        self._worker_started = True

        async def _worker():
            loop = asyncio.get_running_loop()
            while self._running:
                try:
                    # Get from recv queue with timeout
                    msg = await loop.run_in_executor(
                        self.executor,
                        lambda: self.activation_recv_queue.get(timeout=0.1),
                    )
                except pyq.Empty:
                    await asyncio.sleep(0.01)
                    continue
                except Exception:
                    break

                # Track KV initialization for this nonce
                self.get_or_make_kv(msg.nonce)

                # Simulate prefetch delay before compute kicks in
                if self.prefetch_delay_s > 0:
                    await asyncio.sleep(self.prefetch_delay_s)

                # Simulate compute delay
                if self.compute_delay_s > 0:
                    await asyncio.sleep(self.compute_delay_s)

                # Record timestamp
                self.timestamps.append(time.time())

                # Simulate layer processing: update layer_id to last assigned layer
                # This mimics real compute policies that output layer_id = last_processed_layer
                if self._assigned_sorted:
                    # Find the highest assigned layer that we would have processed
                    msg.layer_id = self._assigned_sorted[-1]

                # Attach a tensor view for serialization to avoid output pool misses
                try:
                    buf = self.input_pool.get_buffer(msg.pool_id)
                    if buf is not None:
                        msg.tensor = buf.reshape(msg.shape)
                except Exception:
                    msg.tensor = None

                # Fallback: allocate an output buffer if input view failed
                if msg.tensor is None:
                    try:
                        out_pool_id = self.output_pool.allocate_for_layer(
                            layer_id=msg.layer_id,
                            dtype=self._wire_mx_dtype,
                            shape=msg.shape,
                        )
                        if out_pool_id is not None:
                            msg.pool_id = out_pool_id
                            out_buf = self.output_pool.get_buffer(out_pool_id)
                            if out_buf is not None:
                                msg.tensor = out_buf.reshape(msg.shape)
                    except Exception:
                        msg.tensor = None

                # Enqueue to send queue (simulating output)
                try:
                    msg.is_final = False  # Mock: not final unless specified
                    await loop.run_in_executor(
                        None,
                        self.activation_send_queue.put_nowait,
                        msg,
                    )
                except Exception:
                    pass

        self._worker_task = asyncio.create_task(_worker())

    async def stop_worker(self):
        """Stop the worker."""
        self._running = False
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
        self._worker_started = False

    def close(self):
        """Cleanup executor."""
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
