"""
Halt Monitor V8 - Trading Halt Tracking and Prediction
=======================================================

Monitors and predicts:
- LULD (Limit Up/Limit Down) halts
- Volatility halts (MWCB - Market-Wide Circuit Breakers)
- News pending halts (T1, H10)
- Regulatory halts
- SEC suspension halts

Features:
- Real-time halt detection
- Pre-halt warning system
- Halt probability scoring
- Historical halt patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)


class HaltCode(Enum):
    """SEC/Exchange halt codes."""
    # News-related
    T1 = "T1"      # News pending
    T2 = "T2"      # News released
    T3 = "T3"      # News and resume time not yet determined

    # LULD (Limit Up/Limit Down)
    LUDP = "LUDP"  # LULD pause
    LUDS = "LUDS"  # LULD straddle

    # Regulatory
    T5 = "T5"      # Single stock trading pause (SEC)
    T6 = "T6"      # Extraordinary market activity
    T12 = "T12"    # Additional information requested by exchange

    # Circuit breakers
    M = "M"        # Market-wide circuit breaker (Level 1)
    M1 = "M1"      # MWCB Level 1 (7%)
    M2 = "M2"      # MWCB Level 2 (13%)
    M3 = "M3"      # MWCB Level 3 (20%)

    # IPO related
    H10 = "H10"    # IPO issue not yet trading
    H11 = "H11"    # IPO issue order imbalance

    # Corporate action
    H4 = "H4"      # Non-compliance with listing requirements
    H9 = "H9"      # Not current in required filings
    H2 = "H2"      # Pending halted

    # SEC related
    H1 = "H1"      # Regulatory concern
    H3 = "H3"      # SEC trading suspension

    # Other
    O1 = "O1"      # Operations halt
    UNKNOWN = "UNKNOWN"


class HaltReason(Enum):
    """High-level halt reason categories."""
    NEWS_PENDING = "NEWS_PENDING"
    NEWS_DISSEMINATION = "NEWS_DISSEMINATION"
    VOLATILITY = "VOLATILITY"
    LULD = "LULD"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    REGULATORY = "REGULATORY"
    IPO = "IPO"
    COMPLIANCE = "COMPLIANCE"
    SEC_SUSPENSION = "SEC_SUSPENSION"
    OPERATIONAL = "OPERATIONAL"
    UNKNOWN = "UNKNOWN"


class HaltRisk(Enum):
    """Halt risk probability levels."""
    IMMINENT = "IMMINENT"     # >80% likely in next 5 min
    HIGH = "HIGH"             # >50% likely in next 15 min
    ELEVATED = "ELEVATED"     # >25% likely in next hour
    MODERATE = "MODERATE"     # Some risk factors present
    LOW = "LOW"               # Normal trading conditions


# Map halt codes to reasons
HALT_CODE_REASONS = {
    HaltCode.T1: HaltReason.NEWS_PENDING,
    HaltCode.T2: HaltReason.NEWS_DISSEMINATION,
    HaltCode.T3: HaltReason.NEWS_PENDING,
    HaltCode.LUDP: HaltReason.LULD,
    HaltCode.LUDS: HaltReason.LULD,
    HaltCode.T5: HaltReason.REGULATORY,
    HaltCode.T6: HaltReason.VOLATILITY,
    HaltCode.M: HaltReason.CIRCUIT_BREAKER,
    HaltCode.M1: HaltReason.CIRCUIT_BREAKER,
    HaltCode.M2: HaltReason.CIRCUIT_BREAKER,
    HaltCode.M3: HaltReason.CIRCUIT_BREAKER,
    HaltCode.H10: HaltReason.IPO,
    HaltCode.H11: HaltReason.IPO,
    HaltCode.H4: HaltReason.COMPLIANCE,
    HaltCode.H9: HaltReason.COMPLIANCE,
    HaltCode.H1: HaltReason.REGULATORY,
    HaltCode.H3: HaltReason.SEC_SUSPENSION,
    HaltCode.O1: HaltReason.OPERATIONAL,
}


@dataclass
class HaltEvent:
    """Represents a trading halt event."""
    ticker: str
    halt_code: HaltCode
    reason: HaltReason
    halt_time: datetime

    # Resume info
    resume_time: Optional[datetime] = None
    resume_quote_time: Optional[datetime] = None
    is_resumed: bool = False

    # Details
    exchange: str = ""
    description: str = ""

    # Price context
    halt_price: Optional[float] = None
    resume_price: Optional[float] = None
    luld_lower_band: Optional[float] = None
    luld_upper_band: Optional[float] = None

    def duration_minutes(self) -> Optional[float]:
        """Get halt duration in minutes."""
        if self.resume_time:
            delta = self.resume_time - self.halt_time
            return delta.total_seconds() / 60
        return None

    def is_active(self) -> bool:
        """Check if halt is currently active."""
        return not self.is_resumed and self.resume_time is None


@dataclass
class LULDState:
    """LULD (Limit Up/Limit Down) band state."""
    ticker: str
    reference_price: float
    upper_band: float
    lower_band: float
    band_percent: float  # 5%, 10%, 20%, etc.

    # Current state
    current_price: float = 0.0
    distance_to_upper: float = 0.0  # Percentage
    distance_to_lower: float = 0.0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    bands_updated: datetime = field(default_factory=datetime.now)

    def update_price(self, price: float) -> None:
        """Update current price and distances."""
        self.current_price = price
        self.distance_to_upper = ((self.upper_band - price) / price) * 100
        self.distance_to_lower = ((price - self.lower_band) / price) * 100
        self.last_updated = datetime.now()

    def is_near_limit(self, threshold_pct: float = 1.0) -> bool:
        """Check if price is near LULD limit."""
        return (
            self.distance_to_upper < threshold_pct or
            self.distance_to_lower < threshold_pct
        )


@dataclass
class HaltPrediction:
    """Halt probability prediction."""
    ticker: str
    risk_level: HaltRisk
    probability: float  # 0-100

    # Risk factors
    factors: List[str] = field(default_factory=list)

    # LULD info
    near_luld_limit: bool = False
    luld_state: Optional[LULDState] = None

    # Volatility
    current_volatility: float = 0.0
    volatility_spike: bool = False

    # News
    pending_news: bool = False
    news_embargo_likely: bool = False

    # Timestamps
    predicted_at: datetime = field(default_factory=datetime.now)

    def get_block_reason(self) -> Optional[str]:
        """Get reason string if trade should be blocked."""
        if self.risk_level == HaltRisk.IMMINENT:
            return "HALT_IMMINENT"
        if self.risk_level == HaltRisk.HIGH and self.near_luld_limit:
            return "LULD_RISK"
        return None


@dataclass
class HaltProfile:
    """Complete halt profile for a ticker."""
    ticker: str

    # Current state
    is_halted: bool = False
    current_halt: Optional[HaltEvent] = None
    prediction: Optional[HaltPrediction] = None

    # History
    halt_history: List[HaltEvent] = field(default_factory=list)
    halts_today: int = 0
    halts_this_week: int = 0

    # LULD state
    luld_state: Optional[LULDState] = None

    # Stats
    avg_halt_duration: float = 0.0  # minutes
    frequent_halter: bool = False   # >3 halts per week average

    # Timestamps
    last_halt: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on halt risk."""
        if self.is_halted:
            return 0.0

        if self.prediction:
            multipliers = {
                HaltRisk.IMMINENT: 0.0,
                HaltRisk.HIGH: 0.25,
                HaltRisk.ELEVATED: 0.50,
                HaltRisk.MODERATE: 0.75,
                HaltRisk.LOW: 1.0,
            }
            return multipliers.get(self.prediction.risk_level, 1.0)

        return 1.0 if not self.frequent_halter else 0.75


# LULD band percentages by tier and time
# Tier 1: S&P 500, Russell 1000, select ETFs
# Tier 2: All other NMS securities
LULD_BANDS = {
    "tier1": {
        "normal": 5.0,       # 9:45 AM - 3:35 PM
        "opening": 10.0,     # 9:30 AM - 9:45 AM
        "closing": 10.0,     # 3:35 PM - 4:00 PM
    },
    "tier2": {
        "normal": 10.0,
        "opening": 20.0,
        "closing": 20.0,
    },
    "low_price": {  # Price < $3
        "normal": 20.0,
        "opening": 40.0,
        "closing": 40.0,
    },
}


class HaltMonitor:
    """
    Monitors trading halts and predicts halt probability.

    Usage:
        monitor = HaltMonitor()

        # Check current halt status
        profile = monitor.get_profile(ticker)
        if profile.is_halted:
            wait_for_resume()

        # Get halt prediction
        prediction = monitor.predict_halt(ticker, current_price, volatility)
        if prediction.risk_level == HaltRisk.IMMINENT:
            avoid_new_positions()

        # Update on halt event
        monitor.record_halt(ticker, halt_code, halt_time)
    """

    def __init__(self):
        # Halt profiles by ticker
        self._profiles: Dict[str, HaltProfile] = {}

        # Currently halted tickers
        self._halted: Set[str] = set()

        # LULD states
        self._luld_states: Dict[str, LULDState] = {}

        # Halt event listeners
        self._listeners: List[callable] = []

        # Volatility thresholds for halt prediction
        self._volatility_thresholds = {
            "spike": 150,      # 150% of normal = spike
            "extreme": 300,    # 300% = extreme
        }

    def get_profile(self, ticker: str) -> HaltProfile:
        """Get halt profile for ticker, creating if needed."""
        ticker = ticker.upper()
        if ticker not in self._profiles:
            self._profiles[ticker] = HaltProfile(ticker=ticker)
        return self._profiles[ticker]

    def is_halted(self, ticker: str) -> bool:
        """Quick check if ticker is currently halted."""
        return ticker.upper() in self._halted

    def record_halt(
        self,
        ticker: str,
        halt_code: HaltCode,
        halt_time: Optional[datetime] = None,
        halt_price: Optional[float] = None,
        description: str = "",
        exchange: str = ""
    ) -> HaltEvent:
        """Record a new halt event."""
        ticker = ticker.upper()
        halt_time = halt_time or datetime.now()

        reason = HALT_CODE_REASONS.get(halt_code, HaltReason.UNKNOWN)

        event = HaltEvent(
            ticker=ticker,
            halt_code=halt_code,
            reason=reason,
            halt_time=halt_time,
            halt_price=halt_price,
            description=description,
            exchange=exchange
        )

        # Update profile
        profile = self.get_profile(ticker)
        profile.is_halted = True
        profile.current_halt = event
        profile.halt_history.append(event)
        profile.last_halt = halt_time
        profile.last_updated = datetime.now()

        # Update halts today count
        today = datetime.now().date()
        profile.halts_today = sum(
            1 for h in profile.halt_history
            if h.halt_time.date() == today
        )

        # Update halts this week
        week_ago = datetime.now() - timedelta(days=7)
        profile.halts_this_week = sum(
            1 for h in profile.halt_history
            if h.halt_time > week_ago
        )

        # Check if frequent halter
        profile.frequent_halter = profile.halts_this_week >= 3

        # Add to halted set
        self._halted.add(ticker)

        # Notify listeners
        self._notify_halt(event)

        logger.info(f"HALT recorded: {ticker} - {halt_code.value} at {halt_time}")

        return event

    def record_resume(
        self,
        ticker: str,
        resume_time: Optional[datetime] = None,
        resume_price: Optional[float] = None
    ) -> Optional[HaltEvent]:
        """Record halt resume."""
        ticker = ticker.upper()
        resume_time = resume_time or datetime.now()

        profile = self.get_profile(ticker)

        if not profile.current_halt:
            logger.warning(f"Resume recorded but no active halt for {ticker}")
            return None

        # Update halt event
        profile.current_halt.resume_time = resume_time
        profile.current_halt.resume_price = resume_price
        profile.current_halt.is_resumed = True

        # Update profile
        profile.is_halted = False
        event = profile.current_halt
        profile.current_halt = None
        profile.last_updated = datetime.now()

        # Update average duration
        durations = [
            h.duration_minutes() for h in profile.halt_history
            if h.duration_minutes() is not None
        ]
        if durations:
            profile.avg_halt_duration = sum(durations) / len(durations)

        # Remove from halted set
        self._halted.discard(ticker)

        logger.info(f"RESUME recorded: {ticker} at {resume_time}")

        return event

    def update_luld_bands(
        self,
        ticker: str,
        reference_price: float,
        upper_band: float,
        lower_band: float,
        tier: str = "tier2"
    ) -> LULDState:
        """Update LULD bands for ticker."""
        ticker = ticker.upper()

        band_pct = ((upper_band - reference_price) / reference_price) * 100

        state = LULDState(
            ticker=ticker,
            reference_price=reference_price,
            upper_band=upper_band,
            lower_band=lower_band,
            band_percent=band_pct,
            current_price=reference_price
        )

        self._luld_states[ticker] = state

        # Update profile
        profile = self.get_profile(ticker)
        profile.luld_state = state

        return state

    def update_price(
        self,
        ticker: str,
        price: float,
        volume: Optional[int] = None
    ) -> Optional[LULDState]:
        """Update current price for LULD tracking."""
        ticker = ticker.upper()

        if ticker in self._luld_states:
            state = self._luld_states[ticker]
            state.update_price(price)
            return state

        return None

    def predict_halt(
        self,
        ticker: str,
        current_price: Optional[float] = None,
        volatility: Optional[float] = None,
        normal_volatility: Optional[float] = None,
        has_pending_news: bool = False,
        recent_price_change_pct: Optional[float] = None
    ) -> HaltPrediction:
        """
        Predict halt probability for ticker.

        Args:
            ticker: Stock ticker
            current_price: Current price
            volatility: Current volatility measure
            normal_volatility: Baseline volatility for comparison
            has_pending_news: Whether news is expected
            recent_price_change_pct: Recent price change percentage

        Returns:
            HaltPrediction with probability and risk factors
        """
        ticker = ticker.upper()
        factors = []
        probability = 0.0

        profile = self.get_profile(ticker)
        luld_state = self._luld_states.get(ticker)

        # Check if already halted
        if profile.is_halted:
            prediction = HaltPrediction(
                ticker=ticker,
                risk_level=HaltRisk.IMMINENT,
                probability=100.0,
                factors=["CURRENTLY_HALTED"]
            )
            profile.prediction = prediction
            return prediction

        # Factor 1: LULD proximity
        near_luld = False
        if luld_state and current_price:
            luld_state.update_price(current_price)

            if luld_state.is_near_limit(0.5):  # Within 0.5%
                probability += 50
                factors.append(f"PRICE_AT_LULD_LIMIT ({luld_state.distance_to_upper:.1f}% to upper)")
                near_luld = True
            elif luld_state.is_near_limit(1.0):  # Within 1%
                probability += 30
                factors.append(f"PRICE_NEAR_LULD ({luld_state.distance_to_upper:.1f}% to upper)")
                near_luld = True
            elif luld_state.is_near_limit(2.0):  # Within 2%
                probability += 15
                factors.append("PRICE_APPROACHING_LULD")

        # Factor 2: Volatility spike
        volatility_spike = False
        if volatility and normal_volatility and normal_volatility > 0:
            vol_ratio = volatility / normal_volatility
            if vol_ratio >= 3.0:  # 300%+ = extreme
                probability += 35
                factors.append(f"EXTREME_VOLATILITY ({vol_ratio:.1f}x normal)")
                volatility_spike = True
            elif vol_ratio >= 1.5:  # 150%+ = spike
                probability += 20
                factors.append(f"VOLATILITY_SPIKE ({vol_ratio:.1f}x normal)")
                volatility_spike = True

        # Factor 3: Rapid price movement
        if recent_price_change_pct:
            abs_change = abs(recent_price_change_pct)
            if abs_change >= 20:
                probability += 30
                factors.append(f"RAPID_PRICE_MOVE ({recent_price_change_pct:+.1f}%)")
            elif abs_change >= 10:
                probability += 15
                factors.append(f"SIGNIFICANT_PRICE_MOVE ({recent_price_change_pct:+.1f}%)")

        # Factor 4: Pending news
        if has_pending_news:
            probability += 25
            factors.append("PENDING_NEWS")

        # Factor 5: Historical halt frequency
        if profile.frequent_halter:
            probability += 15
            factors.append(f"FREQUENT_HALTER ({profile.halts_this_week} this week)")

        if profile.halts_today >= 2:
            probability += 10
            factors.append(f"MULTIPLE_HALTS_TODAY ({profile.halts_today})")

        # Factor 6: Recent halt (might resume and halt again)
        if profile.last_halt:
            minutes_since = (datetime.now() - profile.last_halt).total_seconds() / 60
            if minutes_since < 30:
                probability += 15
                factors.append(f"RECENT_HALT ({minutes_since:.0f} min ago)")

        # Cap probability
        probability = min(100.0, probability)

        # Determine risk level
        if probability >= 80:
            risk_level = HaltRisk.IMMINENT
        elif probability >= 50:
            risk_level = HaltRisk.HIGH
        elif probability >= 25:
            risk_level = HaltRisk.ELEVATED
        elif probability >= 10:
            risk_level = HaltRisk.MODERATE
        else:
            risk_level = HaltRisk.LOW

        prediction = HaltPrediction(
            ticker=ticker,
            risk_level=risk_level,
            probability=probability,
            factors=factors,
            near_luld_limit=near_luld,
            luld_state=luld_state,
            current_volatility=volatility or 0.0,
            volatility_spike=volatility_spike,
            pending_news=has_pending_news
        )

        profile.prediction = prediction
        profile.last_updated = datetime.now()

        return prediction

    def get_halted_tickers(self) -> Set[str]:
        """Get set of currently halted tickers."""
        return self._halted.copy()

    def get_high_risk_tickers(self) -> List[Tuple[str, HaltRisk]]:
        """Get tickers with elevated halt risk."""
        result = []
        for ticker, profile in self._profiles.items():
            if profile.prediction:
                if profile.prediction.risk_level in [
                    HaltRisk.IMMINENT,
                    HaltRisk.HIGH,
                    HaltRisk.ELEVATED
                ]:
                    result.append((ticker, profile.prediction.risk_level))
        return sorted(result, key=lambda x: x[1].value)

    def add_halt_listener(self, callback: callable) -> None:
        """Add callback for halt events."""
        self._listeners.append(callback)

    def remove_halt_listener(self, callback: callable) -> None:
        """Remove halt event callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_halt(self, event: HaltEvent) -> None:
        """Notify listeners of halt event."""
        for callback in self._listeners:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in halt listener: {e}")

    def calculate_luld_bands(
        self,
        reference_price: float,
        tier: str = "tier2",
        market_phase: str = "normal"
    ) -> Tuple[float, float]:
        """
        Calculate LULD bands for a price.

        Args:
            reference_price: Reference price for bands
            tier: "tier1", "tier2", or "low_price"
            market_phase: "normal", "opening", or "closing"

        Returns:
            Tuple of (lower_band, upper_band)
        """
        # Determine tier based on price if low
        if reference_price < 3.0:
            tier = "low_price"
        elif reference_price < 0.75:
            # Below $0.75 has wider bands
            tier = "low_price"

        bands = LULD_BANDS.get(tier, LULD_BANDS["tier2"])
        band_pct = bands.get(market_phase, bands["normal"]) / 100

        # Calculate bands (doubled for prices < $0.75)
        if reference_price < 0.75:
            band_pct = min(band_pct * 2, 0.75)  # Cap at 75%

        upper = reference_price * (1 + band_pct)
        lower = reference_price * (1 - band_pct)

        # Round to tick size
        if reference_price >= 1.0:
            upper = round(upper, 2)
            lower = round(lower, 2)
        else:
            upper = round(upper, 4)
            lower = round(lower, 4)

        return (lower, upper)

    def clear_history(self, ticker: Optional[str] = None, days: int = 30) -> None:
        """Clear old halt history."""
        cutoff = datetime.now() - timedelta(days=days)

        if ticker:
            profile = self._profiles.get(ticker.upper())
            if profile:
                profile.halt_history = [
                    h for h in profile.halt_history
                    if h.halt_time > cutoff
                ]
        else:
            for profile in self._profiles.values():
                profile.halt_history = [
                    h for h in profile.halt_history
                    if h.halt_time > cutoff
                ]


# Singleton instance
_monitor: Optional[HaltMonitor] = None


def get_halt_monitor() -> HaltMonitor:
    """Get singleton HaltMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = HaltMonitor()
    return _monitor
