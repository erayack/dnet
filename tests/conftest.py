"""Shared test fixtures and helpers to reduce duplication."""

import asyncio
import typing as _t
import pytest


@pytest.fixture
def wait_until():
    """Async poller to replace arbitrary sleeps in tests.

    Usage:
      await wait_until(lambda: condition(), timeout=0.5)
    """

    async def _wait(
        cond: _t.Callable[[], bool], timeout: float = 0.5, interval: float = 0.01
    ) -> bool:
        deadline = asyncio.get_event_loop().time() + float(timeout)
        while True:
            try:
                if cond():
                    return True
            except Exception:
                # Treat exceptions as not-ready; keep polling
                pass
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(interval)

    return _wait


@pytest.fixture
def patch_http_grpc_client(monkeypatch):
    """Patch shard HTTP gRPC client calls to use FakeChannel/FakeRingStub.

    Returns a function that accepts a dict to capture the last dialed address.
    """

    def _apply(seen_addr: dict):
        from tests.fakes import FakeChannel, FakeRingStub

        def _fake_ch(address: str):
            seen_addr["addr"] = address
            return FakeChannel(address)

        monkeypatch.setattr(
            "dnet.shard.http_api.aio_grpc.insecure_channel", _fake_ch, raising=True
        )
        monkeypatch.setattr(
            "dnet.shard.http_api.DnetRingServiceStub",
            lambda ch: FakeRingStub(ch),
            raising=True,
        )
        return seen_addr

    return _apply


@pytest.fixture
def shard_http_server():
    """Create a Shard HTTPServer + FakeShard with routes set up.

    Provides (srv, shard) and ensures consistent instance name 'S1'.
    """
    import asyncio
    from tests.fakes import FakeShard, FakeDiscovery
    from dnet.shard.http_api import HTTPServer

    shard = FakeShard(1)
    disc = FakeDiscovery({})
    setattr(disc, "instance_name", lambda: "S1")
    srv = HTTPServer(http_port=0, grpc_port=9001, shard=shard, discovery=disc)
    asyncio.run(srv._setup_routes())
    return srv, shard


@pytest.fixture
def patch_shard_hypercorn_serve(monkeypatch):
    """Patch shard HTTPServer's hypercorn serve entry with a controllable fake.

    Returns a dict started={'ok': bool} and the fake function is installed.
    """
    started = {"ok": False}

    async def fake_serve(app, config, shutdown_trigger):
        started["ok"] = True
        fut = shutdown_trigger()
        try:
            await fut
        except Exception:
            pass
        return

    monkeypatch.setattr(
        "dnet.shard.http_api.aio_hypercorn.serve", fake_serve, raising=True
    )
    return started


@pytest.fixture
def patch_api_hypercorn_serve(monkeypatch):
    """Patch API HTTPServer's hypercorn serve entry with a controllable fake.

    Returns a dict started={'ok': bool} and the fake function is installed.
    """
    started = {"ok": False}

    async def fake_serve(app, config, shutdown_trigger):
        started["ok"] = True
        fut = shutdown_trigger()
        try:
            await fut
        except Exception:
            pass
        return

    monkeypatch.setattr(
        "dnet.api.http_api.aio_hypercorn.serve", fake_serve, raising=True
    )
    return started


@pytest.fixture
def patch_ring_grpc_client_ok(monkeypatch):
    """Patch RingAdapter gRPC client to use FakeChannel/FakeRingStub and capture dialed address."""

    def _apply(seen_addr: dict):
        from tests.fakes import FakeChannel, FakeRingStub

        monkeypatch.setattr(
            "dnet.shard.adapters.ring.aio_grpc.insecure_channel",
            lambda addr: (seen_addr.__setitem__("addr", addr) or FakeChannel(addr)),
            raising=True,
        )
        monkeypatch.setattr(
            "dnet.shard.adapters.ring.DnetRingServiceStub",
            lambda ch: FakeRingStub(ch),
            raising=True,
        )
        return seen_addr

    return _apply


@pytest.fixture
def loopback_ring_pipeline(monkeypatch):
    """Wire N RingAdapters in a loopback ring for pipelined testing.

    Returns a function that accepts:
        - n_nodes: number of nodes in the ring
        - compute_delays: list of compute_delay_s for each node
        - assigned_layers: list of sets of layer IDs for each node

    Returns: (adapters, runtimes, delivery_log_dict)
    """

    def _build_pipeline(
        n_nodes: int = 3,
        compute_delays: list[float] = None,
        assigned_layers: list[set] = None,
    ):
        from tests.fakes import (
            FakeDiscovery,
            FakeChannel,
            SlowFakeRuntime,
            LoopbackRingStub,
            SimulatedStreamCtx,
        )
        from dnet.shard.adapters.ring import RingAdapter
        from dnet.shard.config import TransportConfig
        from collections import defaultdict

        if compute_delays is None:
            compute_delays = [0.0] * n_nodes
        if assigned_layers is None:
            assigned_layers = [set() for _ in range(n_nodes)]

        # Create runtimes and adapters
        runtimes = []
        adapters = []
        delivery_log = defaultdict(list)

        for i in range(n_nodes):
            rt = SlowFakeRuntime(
                shard_id=f"S{i}",
                max_queue_size=16,
                assigned_next=assigned_layers[i],
                compute_delay_s=compute_delays[i],
            )
            disc = FakeDiscovery({})
            cfg = TransportConfig(streaming=True)
            ad = RingAdapter(runtime=rt, discovery=disc, transport_config=cfg)
            runtimes.append(rt)
            adapters.append(ad)

        # Wire them in a ring via loopback stubs
        for i in range(n_nodes):
            next_idx = (i + 1) % n_nodes
            target_adapter = adapters[next_idx]

            # Create loopback stub with shared delivery log
            loopback_stub = LoopbackRingStub(target_adapter, shared_log=delivery_log)

            # Patch the adapter to use loopback stub
            adapters[i].next_node_stub = loopback_stub

            # Create fake channel
            fake_channel = FakeChannel(f"loopback-{i}-to-{next_idx}")
            adapters[i].next_node_channel = fake_channel

            # Patch _streams.get_or_create_stream to return SimulatedStreamCtx
            async def _patched_get_or_create(nonce, call_factory, _ad=adapters[i]):
                existing = _ad._streams.get_ctx(nonce)
                if existing and getattr(existing, "open", False):
                    try:
                        loop = asyncio.get_running_loop()
                        if existing.disabled and loop.time() >= existing.disabled_until:
                            existing.disabled = False
                    except Exception:
                        pass
                    return existing

                if existing:
                    try:
                        existing.open = False
                        try:
                            existing.queue.put_nowait(None)
                        except asyncio.QueueFull:
                            await existing.queue.put(None)
                    except Exception:
                        pass
                    try:
                        if existing.ack_task:
                            existing.ack_task.cancel()
                    except Exception:
                        pass

                # Create simulated context
                ctx = SimulatedStreamCtx(nonce)
                _ad._streams._streams[nonce] = ctx

                # Start the call_factory with the context's queue as iterator
                async def _req_iter():
                    while True:
                        item = await ctx.queue.get()
                        if item is None:
                            break
                        yield item

                ctx.call = call_factory(_req_iter())
                ctx.open = True
                ctx.last_activity_t = asyncio.get_running_loop().time()

                # Drive the async generator to pump frames through the loopback stub
                async def _consume_responses():
                    try:
                        async for ack in ctx.call:
                            accepted = getattr(ack, "accepted", None)
                            if accepted is None:
                                accepted = True
                            ctx.acks.append(
                                (
                                    getattr(ack, "seq", None),
                                    accepted,
                                )
                            )
                    except Exception:
                        # Stream closed or error - normal in tests
                        pass
                    finally:
                        ctx.open = False

                # Start the consumer task to drive frame delivery
                ctx.ack_task = asyncio.create_task(_consume_responses())

                return ctx

            adapters[i]._streams.get_or_create_stream = _patched_get_or_create

        return adapters, runtimes, delivery_log

    return _build_pipeline


@pytest.fixture
def make_activation_request():
    """Helper to build ActivationRequest/ActivationMessage pairs for tests.

    Returns a function that accepts:
        - nonce: request nonce
        - layer_id: layer identifier
        - shape: tensor shape (default: (1, 512))
        - dtype: data type (default: "float16")
        - req_top_logprobs: top logprobs count (default: 0)
        - callback_url: callback URL (default: "")

    Returns: ActivationRequest proto
    """

    def _build(
        nonce: str,
        layer_id: int,
        shape: tuple = (1, 512),
        dtype: str = "float16",
        req_top_logprobs: int = 0,
        callback_url: str = "",
    ):
        import numpy as np
        from dnet.protos.dnet_ring_pb2 import Activation, ActivationRequest

        # Create small zeros buffer to avoid memory issues
        if dtype == "float16":
            data = np.zeros(shape, dtype=np.float16).tobytes()
        elif dtype == "float32":
            data = np.zeros(shape, dtype=np.float32).tobytes()
        else:
            data = np.zeros(shape, dtype=np.float32).tobytes()

        activation = Activation(
            data=data,
            batch_size=shape[0],
            shape=list(shape),
            dtype=dtype,
            layer_id=layer_id,
        )

        request = ActivationRequest(
            nonce=nonce,
            activation=activation,
            timestamp=int(asyncio.get_event_loop().time() * 1000),
            node_origin="test",
            callback_url=callback_url,
            logprobs=False,
            top_logprobs=req_top_logprobs,
        )

        return request

    return _build
