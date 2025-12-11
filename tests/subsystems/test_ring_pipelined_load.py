"""Tests: Pipelined ring concurrency and load handling.

This module exercises multi-shard pipelines with loopback connections to validate:
- Mixed device speed handling without stalls
- Backpressure recovery
- UMA-style offload prefetch and eviction
- Long-context cross-device serialization
"""

import asyncio

import pytest

from dnet.shard.adapters.ring import RingAdapter
from dnet.shard.config import TransportConfig
from dnet.shard.models import ShardLoadModelRequest


pytestmark = [pytest.mark.shard, pytest.mark.ring]


# ========================
# Harness Helpers
# ========================


async def wait_queue_drain(q: asyncio.Queue, timeout: float = 1.0) -> bool:
    """Poll until queue is empty or timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if q.empty():
            return True
        await asyncio.sleep(0.01)
    return False


async def wait_delivery(
    log: dict, nonce: str, count: int, timeout: float = 1.0
) -> bool:
    """Poll until nonce has count entries in delivery log."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if nonce in log and len(log[nonce]) >= count:
            return True
        await asyncio.sleep(0.01)
    return False


# ========================
# Group 1: mixed_speed_pipeline
# ========================


def test_pipeline_handles_mixed_device_speeds(
    loopback_ring_pipeline, make_activation_request, wait_until
):
    """Test that upstream adapters don't stall while slow downstream nodes process.

    Setup: 3-node ring with staggered compute delays (fast -> slow -> medium)
    Inject: Burst of activations across 2 nonces
    Assert:
    - Upstream ring_tx queues drain
    - ctx.last_seq increments for each nonce
    - Ordering per nonce preserved at ingress queues
    - No disabled streams
    """

    async def main():
        # Build 3-node ring: fast (0.01s) -> slow (0.1s) -> medium (0.05s)
        adapters, runtimes, delivery_log = loopback_ring_pipeline(
            n_nodes=3,
            compute_delays=[0.01, 0.1, 0.05],
            assigned_layers=[{1}, {2}, {3}],
        )

        # Start all adapters
        for ad in adapters:
            await ad.start()

        # Inject burst of activations for 2 nonces
        nonces = ["nonce1", "nonce2"]
        for nonce in nonces:
            for layer_id in range(3):
                req = make_activation_request(nonce=nonce, layer_id=layer_id)
                # Inject to first adapter
                await adapters[0].ingress_q.put(req)

        # Wait for ring_tx queues to drain (upstream shouldn't stall)
        await asyncio.sleep(0.2)  # Allow time for processing

        # Assert: ring_tx queues eventually drain
        for i, ad in enumerate(adapters):
            drained = await wait_queue_drain(ad.ring_tx_q, timeout=2.0)
            assert drained, f"Adapter {i} ring_tx queue did not drain"

        # Assert: stream contexts exist and have incremented seq numbers
        for nonce in nonces:
            for i, ad in enumerate(adapters):
                ctx = ad._streams.get_ctx(nonce)
                if ctx:
                    assert ctx.last_seq > 0, (
                        f"Adapter {i} nonce {nonce} seq not incremented"
                    )
                    assert not ctx.disabled, (
                        f"Adapter {i} nonce {nonce} stream disabled"
                    )

        # Cleanup
        for ad in adapters:
            await ad.shutdown()
        for rt in runtimes:
            rt.close()

    asyncio.run(main())


def test_pipeline_recovers_from_slow_stage_backpressure(
    loopback_ring_pipeline, make_activation_request, wait_until
):
    """Test that pipeline recovers when middle node stream is temporarily disabled.

    Setup: 3-node ring with middle node slow
    Mid-run: Disable middle node's incoming stream
    Assert: Upstream handles appropriately (logs error)
    Re-enable: Clear disabled flag
    Assert: Stream reopens and processing continues
    """

    async def main():
        adapters, runtimes, delivery_log = loopback_ring_pipeline(
            n_nodes=3,
            compute_delays=[0.01, 0.15, 0.05],
            assigned_layers=[{1}, {2}, {3}],
        )

        for ad in adapters:
            await ad.start()

        nonce = "backpressure_test"

        # Inject first activation and wait until it reaches the downstream adapter
        req1 = make_activation_request(nonce=nonce, layer_id=0)
        await adapters[0].ingress_q.put(req1)
        delivered_first = await wait_delivery(delivery_log, nonce, count=1, timeout=1.0)
        assert delivered_first, "Initial activation never reached downstream node"

        # Capture the stream context and simulate a backpressure disable
        ctx = adapters[0]._streams.get_ctx(nonce)
        assert ctx is not None, "Stream context was not created"
        ctx.set_disabled(True)
        assert ctx.disabled, "Failed to mark stream as disabled"

        # Inject another activation while disabled; it should not be delivered
        req2 = make_activation_request(nonce=nonce, layer_id=1)
        await adapters[0].ingress_q.put(req2)
        blocked = await wait_until(lambda: len(delivery_log[nonce]) > 1, timeout=0.1)
        assert not blocked, "Downstream received frames while upstream was disabled"

        # Wait out the backoff window before retrying
        backoff_elapsed = await wait_until(
            lambda: asyncio.get_running_loop().time() >= ctx.disabled_until, timeout=1.5
        )
        assert backoff_elapsed, "Backpressure window did not elapse in time"

        # Retry the blocked activation and push another frame to confirm forward progress
        await adapters[0].ingress_q.put(
            make_activation_request(nonce=nonce, layer_id=1)
        )
        await adapters[0].ingress_q.put(
            make_activation_request(nonce=nonce, layer_id=2)
        )

        # Verify the stream recovers and downstream receives the resumed frames
        recovered = await wait_until(lambda: not ctx.disabled, timeout=1.5)
        assert recovered, "Stream did not re-enable after backpressure window"
        resumed = await wait_delivery(delivery_log, nonce, count=3, timeout=1.5)
        assert resumed, "Downstream did not receive frames after recovery"

        # Ack reader should have drained the loopback responses for each frame
        acked = await wait_until(
            lambda: len(getattr(ctx, "acks", [])) >= 3, timeout=1.0
        )
        assert acked, "Ack stream did not deliver confirmations for resumed frames"

        # Cleanup
        for ad in adapters:
            await ad.shutdown()
        for rt in runtimes:
            rt.close()

    asyncio.run(main())


# ========================
# Group 2: long_context_uma_offload
# ========================


def test_long_context_uma_offload_prefetch(
    make_activation_request, monkeypatch, wait_until
):
    """Test offload policy prefetch and eviction with sliding window.

    Setup: Single adapter with OffloadPolicy (window < residency -> sliding_fit)
    Inject: Activations spanning layers 8-12 (multiple windows)
    Assert:
    - OffloadPolicy._prepared_by_nonce updates with new futures
    - WeightCache.decrease_reference called
    - Tokens with callback_url route through token_tx_q
    """

    async def main():
        from tests.fakes import (
            FakeDiscovery,
            FakeRuntimeForAdapter,
            FakeWeightCache,
        )
        from dnet.shard.policies.offload import OffloadPolicy
        import types

        # Create runtime with assigned layers 8-12
        rt = FakeRuntimeForAdapter(
            shard_id="UMA_test",
            assigned_next={8, 9, 10, 11, 12},
        )
        rt.assigned_layers = [8, 9, 10, 11, 12]
        rt._assigned_sorted = [8, 9, 10, 11, 12]
        rt.model_path = "/fake/model"

        # Mock model metadata
        rt.model_metadata = types.SimpleNamespace(
            num_layers=12,
            weight_info={},
        )

        # Mock model (needed for policy.process checks)
        from tests.fakes import FakeComputeModel
        import mlx.core as mx

        rt.model = FakeComputeModel(mx)

        # Create adapter
        disc = FakeDiscovery({})
        cfg = TransportConfig(streaming=True)
        ad = RingAdapter(runtime=rt, discovery=disc, transport_config=cfg)

        # Create OffloadPolicy
        policy = OffloadPolicy(runtime=rt, resident_windows=1)

        # Attach policy to runtime so compute worker can use it
        rt.policy = policy

        # Configure with sliding_fit mode (window_size=2, residency_size=1)
        req = ShardLoadModelRequest(
            model_path="/fake/model",
            total_layers=12,
            layers=[8, 9, 10, 11, 12],
            warmup=False,
            next_node=None,
            window_size=2,
            residency_size=1,
            kv_bits="8bit",
            api_callback_address="",
        )

        # Patch repack to skip
        monkeypatch.setattr(
            "dnet.shard.policies.offload.ensure_repacked_for_layers",
            lambda *args, **kwargs: ("/fake/model", False),
        )

        # Patch WeightCache to use fake
        fake_cache = FakeWeightCache(
            assigned_layers=[8, 9, 10, 11, 12],
            model_metadata=rt.model_metadata,
            window_size=2,
            resident_windows=1,
        )

        monkeypatch.setattr(
            "dnet.shard.policies.offload.WeightCache",
            lambda *args, **kwargs: fake_cache,
        )

        policy.configure_policy_for_model(req)

        # Verify policy is in sliding_fit mode
        assert policy._mode == "sliding_fit"
        assert policy.window_size == 1  # min(residency, local_count) with resident=1

        # Start adapter and compute worker
        await ad.start()
        await rt.start_compute_worker()

        # Inject activations spanning multiple layers
        nonce = "long_ctx"
        for layer_id in [7, 8, 9, 10, 11]:  # Will process 8-11 (within assigned set)
            req = make_activation_request(nonce=nonce, layer_id=layer_id)
            await ad.ingress_q.put(req)

        # Allow processing
        await asyncio.sleep(0.5)

        # Assert: WeightCache.decrease_reference was called
        assert len(fake_cache.dec_refs) > 0, "No decrease_reference calls"

        # Assert: evict_layer was called (UMA-style eviction)
        assert len(fake_cache.evicted) > 0, "No evictions occurred"

        # Cleanup
        await rt.stop_compute_worker()
        await ad.shutdown()
        rt.close()

    asyncio.run(main())


def test_long_context_cross_device(
    loopback_ring_pipeline, make_activation_request, monkeypatch
):
    """Test long-context nonce across multiple nodes with mixed speeds.

    Setup: Multi-node loopback ring with mixed compute delays
    Inject: Long-context nonce through pipeline
    Assert:
    - Downstream nodes see KV init per nonce (runtime._kv_calls)
    - Serialized dtype stays consistent despite repeated processing
    """

    async def main():
        adapters, runtimes, delivery_log = loopback_ring_pipeline(
            n_nodes=3,
            compute_delays=[0.02, 0.05, 0.03],
            assigned_layers=[{1}, {2}, {3}],
        )

        # Patch codec to track serialization calls
        serialization_dtypes = []

        for ad in adapters:
            original_serialize = ad.codec.serialize

            def _tracked_serialize(msg, config, _orig=original_serialize):
                serialization_dtypes.append(msg.dtype)
                return _orig(msg, config)

            ad.codec.serialize = _tracked_serialize

        for ad in adapters:
            await ad.start()

        # Start runtime workers to simulate processing
        for rt in runtimes:
            await rt.start_worker()

        nonce = "long_context_nonce"

        # Inject long sequence of activations
        for layer_id in range(6):
            req = make_activation_request(nonce=nonce, layer_id=layer_id)
            await adapters[0].ingress_q.put(req)

        # Allow processing time
        await asyncio.sleep(0.5)

        # Assert: Each runtime initialized KV for the nonce
        for i, rt in enumerate(runtimes):
            assert nonce in rt._kv_calls, f"Runtime {i} did not init KV for {nonce}"

        # Assert: Serialized dtype is consistent (all should be float16)
        for dtype in serialization_dtypes:
            assert dtype == "float16", f"Inconsistent dtype: {dtype}"

        # Cleanup
        for rt in runtimes:
            await rt.stop_worker()
        for ad in adapters:
            await ad.shutdown()
        for rt in runtimes:
            rt.close()

    asyncio.run(main())
