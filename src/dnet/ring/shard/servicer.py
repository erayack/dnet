"""gRPC servicer for shard node."""

import time
from typing import TYPE_CHECKING

import grpc

from ...protos import shard_api_comm_pb2 as pb2_shard_api
from ...protos.dnet_ring_pb2 import (
    ActivationRequest,
    ActivationResponse,
    HealthRequest,
    HealthResponse,
    LatencyMeasureRequest,
    LatencyMeasureResponse,
    ResetCacheRequest,
    ResetCacheResponse,
)
from ...protos.dnet_ring_pb2_grpc import DnetRingServiceServicer
from ...protos.shard_api_comm_pb2_grpc import ShardApiServiceServicer
from ...utils.logger import logger

if TYPE_CHECKING:
    from .node import RingShardNode


class ShardServicer(DnetRingServiceServicer, ShardApiServiceServicer):
    """gRPC servicer implementation for shard node."""

    def __init__(self, node: "RingShardNode") -> None:
        """Initialize shard servicer.

        Args:
            node: The shard node instance
        """
        self.node = node

    async def SendActivation(
        self,
        request: ActivationRequest,
        context: grpc.aio.ServicerContext,
    ) -> ActivationResponse:
        """Handle incoming activation requests.

        Args:
            request: Activation request from previous node
            context: gRPC context

        Returns:
            Activation response
        """
        try:
            logger.debug(
                f"Node {self.node.node_id} received activation: "
                f"nonce={request.nonce}, layer={request.activation.layer_id}"
            )

            # Process the activation
            await self.node.receive_activation(request)

            return ActivationResponse(
                success=True,
                message="Activation processed successfully",
                node_id=str(self.node.node_id),
            )
        except Exception as e:
            logger.error(f"Error processing activation request: {e}")
            return ActivationResponse(
                success=False,
                message=f"Error: {str(e)}",
                node_id=str(self.node.node_id),
            )

    async def HealthCheck(
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> HealthResponse:
        """Handle health check requests.

        Args:
            request: Health check request
            context: gRPC context

        Returns:
            Health status response
        """
        logger.debug(
            f"Node {self.node.node_id} received health request from "
            f"{request.requester_id}"
        )

        return HealthResponse(
            healthy=self.node.running,
            node_id=str(self.node.node_id),
            assigned_layers=self.node.assigned_layers,
            queue_size=self.node.activation_recv_queue.qsize(),
            active_requests=0,  # TODO: implement proper tracking
        )

    async def ResetCache(
        self,
        request: ResetCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> ResetCacheResponse:
        """Handle reset cache requests.

        Args:
            request: Reset cache request
            context: gRPC context

        Returns:
            Reset cache response
        """
        try:
            logger.debug(f"Node {self.node.node_id} received reset cache request")

            # Reset the cache
            await self.node.reset_cache()

            return ResetCacheResponse(
                success=True,
                message="Cache reset successfully",
            )
        except Exception as e:
            logger.error(f"Error processing reset-cache request: {e}")
            return ResetCacheResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    async def MeasureLatency(
        self,
        request: LatencyMeasureRequest,
        context: grpc.aio.ServicerContext,
    ) -> LatencyMeasureResponse:
        """Handle latency measurement requests.

        Args:
            request: Latency measurement request
            context: gRPC context

        Returns:
            Latency measurement response
        """
        try:
            logger.debug(
                f"Node {self.node.node_id} received latency measurement request from "
                f"{request.requester_id}, payload size: {request.payload_size}"
            )

            # Simply respond with success - the latency is measured by the requester
            return LatencyMeasureResponse(
                success=True,
                message="Latency measurement response",
                node_id=str(self.node.node_id),
                timestamp=int(time.time() * 1000),  # Current timestamp in ms
            )
        except Exception as e:
            logger.error(f"Error processing latency measurement request: {e}")
            return LatencyMeasureResponse(
                success=False,
                message=f"Error: {str(e)}",
                node_id=str(self.node.node_id),
                timestamp=int(time.time() * 1000),
            )

    async def LoadModel(
        self,
        request: pb2_shard_api.LoadModelRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2_shard_api.LoadModelResponse:
        """Handle model loading requests.

        Args:
            request: Load model request
            context: gRPC context

        Returns:
            Load model response
        """
        try:
            logger.info(
                f"Node {self.node.node_id} received load model request: "
                f"model_path={request.model_path}, layers={list(request.layers)}, "
                f"warmup={request.warmup}"
            )

            # Delegate to node's load_model method
            result = await self.node.load_model(
                model_path=request.model_path,
                layers=list(request.layers),
                warmup=request.warmup,
            )

            return pb2_shard_api.LoadModelResponse(
                success=result["success"],
                message=result["message"],
                layers_loaded=result.get("layers_loaded", []),
                load_time_ms=result.get("load_time_ms", 0.0),
            )
        except Exception as e:
            logger.error(f"Error processing load model request: {e}")
            return pb2_shard_api.LoadModelResponse(
                success=False,
                message=f"Error: {str(e)}",
                layers_loaded=[],
                load_time_ms=0.0,
            )

    async def UnloadModel(
        self,
        request: pb2_shard_api.UnloadModelRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2_shard_api.UnloadModelResponse:
        """Handle model unloading requests.

        Args:
            request: Unload model request
            context: gRPC context

        Returns:
            Unload model response
        """
        try:
            logger.info(f"Node {self.node.node_id} received unload model request")

            # Delegate to node's unload_model method
            result = await self.node.unload_model()

            return pb2_shard_api.UnloadModelResponse(
                success=result["success"],
                message=result["message"],
            )
        except Exception as e:
            logger.error(f"Error processing unload model request: {e}")
            return pb2_shard_api.UnloadModelResponse(
                success=False,
                message=f"Error: {str(e)}",
            )
