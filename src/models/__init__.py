"""
GV2-EDGE V7.0 Models
====================

Data types and models for the V7 architecture.

Modules:
- signal_types: Unified signal, order, and execution types
"""

from .signal_types import (
    # Layer 1: Detection
    SignalType,
    PreSpikeState,
    PreHaltState,

    # Layer 2: Order
    OrderSide,
    OrderType,
    TimingStrategy,
    ProposedOrder,

    # Layer 3: Execution
    ExecutionStatus,
    BlockReason,
    ExecutionDecision,

    # Unified
    UnifiedSignal,
)

__all__ = [
    # Layer 1: Detection
    "SignalType",
    "PreSpikeState",
    "PreHaltState",

    # Layer 2: Order
    "OrderSide",
    "OrderType",
    "TimingStrategy",
    "ProposedOrder",

    # Layer 3: Execution
    "ExecutionStatus",
    "BlockReason",
    "ExecutionDecision",

    # Unified
    "UnifiedSignal",
]
