"""
GV2-EDGE V6.1 Boosters
======================

Signal boosting modules for confluence scoring.

Modules:
- insider_boost: SEC Form 4 insider activity scoring
- squeeze_boost: Short interest and squeeze potential
"""

from .insider_boost import (
    InsiderBoostEngine,
    InsiderBoostResult,
    InsiderSignal,
    get_insider_engine,
    quick_insider_check,
    apply_insider_boost,
)

from .squeeze_boost import (
    SqueezeBoostEngine,
    SqueezeBoostResult,
    SqueezeSignal,
    ShortData,
    get_squeeze_engine,
    quick_squeeze_check,
    apply_squeeze_boost,
)

__all__ = [
    # Insider Boost
    "InsiderBoostEngine",
    "InsiderBoostResult",
    "InsiderSignal",
    "get_insider_engine",
    "quick_insider_check",
    "apply_insider_boost",
    # Squeeze Boost
    "SqueezeBoostEngine",
    "SqueezeBoostResult",
    "SqueezeSignal",
    "ShortData",
    "get_squeeze_engine",
    "quick_squeeze_check",
    "apply_squeeze_boost",
]
