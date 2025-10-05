"""Data types for dnet ring topology."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

from ..protos.dnet_ring_pb2 import Activation, ActivationRequest


@dataclass(slots=True)
class ActivationMessage:
    """Internal representation of activation data flowing through the ring."""

    nonce: str
    pool_id: int
    batch_size: int
    shape: Tuple[int, ...]
    dtype: str
    layer_id: int
    timestamp: int
    node_origin: str
    callback_url: str

    @classmethod
    def from_proto(
        cls, proto_msg: ActivationRequest, pool_id: int = 0
    ) -> "ActivationMessage":
        """Create ActivationMessage from protobuf message.

        Args:
            proto_msg: Protobuf ActivationRequest
            pool_id: Memory pool ID for this activation

        Returns:
            ActivationMessage instance
        """
        return cls(
            nonce=proto_msg.nonce,
            pool_id=pool_id,
            batch_size=proto_msg.activation.batch_size,
            shape=tuple(proto_msg.activation.shape),
            dtype=proto_msg.activation.dtype,
            layer_id=proto_msg.activation.layer_id,
            timestamp=proto_msg.timestamp,
            node_origin=proto_msg.node_origin,
            callback_url=proto_msg.callback_url,
        )

    def to_proto(self, data: bytes) -> ActivationRequest:
        """Convert to protobuf ActivationRequest.

        Args:
            data: Serialized tensor data

        Returns:
            Protobuf ActivationRequest
        """
        return ActivationRequest(
            nonce=self.nonce,
            activation=Activation(
                data=data,
                batch_size=self.batch_size,
                shape=list(self.shape),
                layer_id=self.layer_id,
                dtype=self.dtype,
            ),
            timestamp=self.timestamp,
            node_origin=self.node_origin,
            callback_url=self.callback_url,
        )


@dataclass(slots=True)
class WeightRequest:
    """Request for weight prefetching."""

    weight_id: str
    layer_id: int
    priority: int = 0


class PoolStatus(str, Enum):
    """Status of a memory pool buffer."""

    FREE = "free"
    ALLOCATED = "allocated"
    IN_USE = "in_use"


class StopCondition(NamedTuple):
    """Condition for stopping token generation."""

    stop_met: bool
    trim_length: int
