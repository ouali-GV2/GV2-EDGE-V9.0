"""
GV2-EDGE V7.0 Engines
=====================

Core engine modules for the V7 architecture.

Architecture:
1. SIGNAL PRODUCER → Détection pure (illimitée)
2. ORDER COMPUTER → Calcul d'ordres (toujours)
3. EXECUTION GATE → Limites et autorisations

Principe fondamental:
- La détection ne s'arrête JAMAIS
- Les limites n'affectent QUE l'exécution
- Le trader voit TOUS les signaux

Usage:
    from src.engines import (
        SignalProducer,
        OrderComputer,
        ExecutionGate
    )

    # 1. Produce signal (unlimited)
    producer = get_signal_producer()
    signal = await producer.detect(input_data)

    # 2. Compute order (always)
    computer = get_order_computer()
    signal = computer.compute_order(signal, market)

    # 3. Evaluate execution (limits apply here)
    gate = get_execution_gate()
    signal = gate.evaluate(signal, risk_flags)

    # 4. Check result
    if signal.is_executable():
        execute(signal.proposed_order)
    else:
        log_blocked(signal)  # Signal still visible!
"""

from .signal_producer import (
    SignalProducer,
    DetectionInput,
    DetectionResult,
    get_signal_producer,
    quick_detect,
)

from .order_computer import (
    OrderComputer,
    PortfolioContext,
    MarketContext,
    get_order_computer,
    compute_order_for_signal,
)

from .execution_gate import (
    ExecutionGate,
    AccountState,
    RiskFlags,
    MarketState,
    get_execution_gate,
    quick_evaluate,
)

__all__ = [
    # Signal Producer
    "SignalProducer",
    "DetectionInput",
    "DetectionResult",
    "get_signal_producer",
    "quick_detect",

    # Order Computer
    "OrderComputer",
    "PortfolioContext",
    "MarketContext",
    "get_order_computer",
    "compute_order_for_signal",

    # Execution Gate
    "ExecutionGate",
    "AccountState",
    "RiskFlags",
    "MarketState",
    "get_execution_gate",
    "quick_evaluate",
]
