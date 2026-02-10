"""
Monday Prep - Strategic Watchlist Generation

Generates prioritized watchlists for Monday trading:
- Pre-market movers candidates
- Gap play setups
- Earnings plays
- Catalyst-driven opportunities
- Technical breakout candidates
- Sector leaders

Designed to run Sunday evening for Monday readiness.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class WatchlistCategory(Enum):
    """Categories for watchlist organization."""
    MOMENTUM = "MOMENTUM"           # Strong momentum plays
    GAP_UP = "GAP_UP"               # Potential gap up candidates
    GAP_DOWN = "GAP_DOWN"           # Gap down / short candidates
    EARNINGS = "EARNINGS"           # Earnings plays
    CATALYST = "CATALYST"           # News/catalyst driven
    TECHNICAL = "TECHNICAL"         # Technical pattern setups
    SQUEEZE = "SQUEEZE"             # Short squeeze candidates
    SECTOR_LEADER = "SECTOR_LEADER" # Leading their sector
    RECOVERY = "RECOVERY"           # Oversold bounce plays
    AVOID = "AVOID"                 # Do not trade


class WatchlistPriority(Enum):
    """Priority levels for watchlist items."""
    PRIMARY = 1      # Top focus, ready to trade
    SECONDARY = 2    # Good setups, second tier
    MONITOR = 3      # Watch for better entry
    SPECULATIVE = 4  # Higher risk, smaller size


@dataclass
class WatchlistItem:
    """A ticker on the watchlist with analysis."""
    ticker: str
    category: WatchlistCategory
    priority: WatchlistPriority
    added_at: datetime = field(default_factory=datetime.now)

    # Scores (0-100)
    overall_score: float = 0.0
    momentum_score: float = 0.0
    risk_score: float = 0.0  # Higher = more risky

    # Price levels
    current_price: float = 0.0
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Key levels
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    pivot_point: Optional[float] = None

    # Analysis
    thesis: str = ""
    catalysts: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)

    # Timing
    earnings_date: Optional[date] = None
    catalyst_date: Optional[date] = None
    expiry_date: Optional[date] = None  # When setup expires

    # Position sizing
    suggested_size: float = 1.0  # Multiplier (0.5 = half size)
    max_risk_pct: float = 1.0    # Max % of portfolio to risk

    # Metadata
    sector: str = ""
    market_cap: Optional[float] = None
    avg_volume: Optional[int] = None
    float_shares: Optional[int] = None
    short_interest: Optional[float] = None

    # Flags
    is_active: bool = True
    needs_confirmation: bool = False
    pre_market_alert: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "category": self.category.value,
            "priority": self.priority.value,
            "overall_score": self.overall_score,
            "current_price": self.current_price,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "thesis": self.thesis,
            "catalysts": self.catalysts,
            "risk_factors": self.risk_factors,
            "suggested_size": self.suggested_size,
        }


@dataclass
class MondayPlan:
    """Complete trading plan for Monday."""
    created_at: datetime = field(default_factory=datetime.now)
    target_date: date = field(default_factory=date.today)

    # Watchlists by category
    watchlists: Dict[WatchlistCategory, List[WatchlistItem]] = field(
        default_factory=dict
    )

    # Priority lists
    primary_focus: List[str] = field(default_factory=list)  # Top 5-10
    secondary_focus: List[str] = field(default_factory=list)
    avoid_list: List[str] = field(default_factory=list)

    # Market context
    market_bias: str = ""  # "BULLISH", "BEARISH", "NEUTRAL"
    sector_leaders: List[str] = field(default_factory=list)
    sector_laggards: List[str] = field(default_factory=list)

    # Key events
    earnings_today: List[str] = field(default_factory=list)
    earnings_this_week: List[str] = field(default_factory=list)
    economic_events: List[Dict] = field(default_factory=list)

    # Risk management
    max_new_positions: int = 5
    suggested_cash_pct: float = 30.0  # % to keep in cash

    # Notes
    notes: List[str] = field(default_factory=list)

    def get_all_tickers(self) -> List[str]:
        """Get all tickers in the plan."""
        tickers = set()
        for items in self.watchlists.values():
            for item in items:
                tickers.add(item.ticker)
        return list(tickers)

    def get_by_priority(self, priority: WatchlistPriority) -> List[WatchlistItem]:
        """Get all items with given priority."""
        result = []
        for items in self.watchlists.values():
            for item in items:
                if item.priority == priority:
                    result.append(item)
        return sorted(result, key=lambda x: x.overall_score, reverse=True)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "target_date": self.target_date.isoformat(),
            "primary_focus": self.primary_focus,
            "secondary_focus": self.secondary_focus,
            "avoid_list": self.avoid_list,
            "market_bias": self.market_bias,
            "sector_leaders": self.sector_leaders,
            "max_new_positions": self.max_new_positions,
            "watchlists": {
                cat.value: [item.to_dict() for item in items]
                for cat, items in self.watchlists.items()
            },
        }


@dataclass
class MondayPrepConfig:
    """Configuration for Monday prep."""
    # Watchlist limits
    max_primary: int = 10
    max_secondary: int = 20
    max_per_category: int = 15

    # Score thresholds
    min_score_primary: float = 75.0
    min_score_secondary: float = 60.0
    min_score_include: float = 50.0

    # Risk limits
    max_risk_score: float = 70.0  # Exclude if higher
    require_stop_loss: bool = True

    # Filters
    min_price: float = 1.0
    max_price: float = 500.0
    min_volume: int = 100_000
    min_float: int = 1_000_000

    # Categories to generate
    categories: List[WatchlistCategory] = field(default_factory=lambda: [
        WatchlistCategory.MOMENTUM,
        WatchlistCategory.GAP_UP,
        WatchlistCategory.EARNINGS,
        WatchlistCategory.CATALYST,
        WatchlistCategory.TECHNICAL,
        WatchlistCategory.SQUEEZE,
    ])


class MondayPrep:
    """
    Generates strategic watchlists for Monday trading.

    Usage:
        prep = MondayPrep()

        # Add candidates from scanner
        for result in scanner.get_candidates():
            prep.add_candidate(result)

        # Generate Monday plan
        plan = prep.generate_plan()

        # Export
        print(plan.primary_focus)
    """

    def __init__(self, config: Optional[MondayPrepConfig] = None):
        self.config = config or MondayPrepConfig()

        # All candidates
        self._candidates: Dict[str, WatchlistItem] = {}

        # Category assignments
        self._by_category: Dict[WatchlistCategory, List[str]] = {
            cat: [] for cat in WatchlistCategory
        }

        # Current plan
        self._current_plan: Optional[MondayPlan] = None

        # Historical plans
        self._plans: Dict[date, MondayPlan] = {}

    def add_candidate(
        self,
        ticker: str,
        category: WatchlistCategory,
        overall_score: float,
        momentum_score: float = 0.0,
        risk_score: float = 0.0,
        current_price: float = 0.0,
        thesis: str = "",
        catalysts: Optional[List[str]] = None,
        risk_factors: Optional[List[str]] = None,
        signals: Optional[List[str]] = None,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> WatchlistItem:
        """
        Add a candidate to the prep list.

        Args:
            ticker: Stock ticker
            category: Watchlist category
            overall_score: Overall score (0-100)
            momentum_score: Momentum score (0-100)
            risk_score: Risk score (0-100, higher = riskier)
            current_price: Current price
            thesis: Investment thesis
            catalysts: List of catalysts
            risk_factors: List of risk factors
            signals: List of signals
            entry_price: Suggested entry price
            target_price: Price target
            stop_price: Stop loss price
            **kwargs: Additional fields

        Returns:
            Created WatchlistItem
        """
        ticker = ticker.upper()

        # Determine priority from score
        if overall_score >= self.config.min_score_primary:
            priority = WatchlistPriority.PRIMARY
        elif overall_score >= self.config.min_score_secondary:
            priority = WatchlistPriority.SECONDARY
        else:
            priority = WatchlistPriority.MONITOR

        # Increase priority if speculative/risky
        if risk_score > 60:
            priority = WatchlistPriority.SPECULATIVE

        item = WatchlistItem(
            ticker=ticker,
            category=category,
            priority=priority,
            overall_score=overall_score,
            momentum_score=momentum_score,
            risk_score=risk_score,
            current_price=current_price,
            thesis=thesis,
            catalysts=catalysts or [],
            risk_factors=risk_factors or [],
            signals=signals or [],
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            **kwargs
        )

        # Calculate suggested size based on risk
        if risk_score > 70:
            item.suggested_size = 0.25
        elif risk_score > 50:
            item.suggested_size = 0.50
        elif risk_score > 30:
            item.suggested_size = 0.75
        else:
            item.suggested_size = 1.0

        # Store
        self._candidates[ticker] = item
        if ticker not in self._by_category[category]:
            self._by_category[category].append(ticker)

        return item

    def add_to_avoid(
        self,
        ticker: str,
        reason: str
    ) -> WatchlistItem:
        """Add ticker to avoid list."""
        return self.add_candidate(
            ticker=ticker,
            category=WatchlistCategory.AVOID,
            overall_score=0.0,
            risk_score=100.0,
            thesis=f"AVOID: {reason}",
            risk_factors=[reason]
        )

    def generate_plan(
        self,
        target_date: Optional[date] = None,
        market_bias: str = "NEUTRAL"
    ) -> MondayPlan:
        """
        Generate Monday trading plan.

        Args:
            target_date: Target trading date (default: next trading day)
            market_bias: Overall market bias

        Returns:
            Complete MondayPlan
        """
        target_date = target_date or self._get_next_trading_day()

        plan = MondayPlan(
            target_date=target_date,
            market_bias=market_bias
        )

        # Build watchlists by category
        for category in self.config.categories:
            items = self._build_category_list(category)
            if items:
                plan.watchlists[category] = items

        # Build avoid list
        avoid_items = self._by_category.get(WatchlistCategory.AVOID, [])
        plan.avoid_list = avoid_items[:50]  # Limit avoid list

        # Select primary focus (top candidates across categories)
        all_items = []
        for items in plan.watchlists.values():
            all_items.extend(items)

        # Sort by score and priority
        all_items.sort(
            key=lambda x: (x.priority.value, -x.overall_score)
        )

        # Primary: top N by score with PRIMARY priority
        primary_candidates = [
            item for item in all_items
            if item.priority == WatchlistPriority.PRIMARY
        ]
        primary_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        plan.primary_focus = [
            item.ticker for item in primary_candidates[:self.config.max_primary]
        ]

        # Secondary: next tier
        secondary_candidates = [
            item for item in all_items
            if item.priority == WatchlistPriority.SECONDARY
            and item.ticker not in plan.primary_focus
        ]
        secondary_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        plan.secondary_focus = [
            item.ticker for item in secondary_candidates[:self.config.max_secondary]
        ]

        # Store plan
        self._current_plan = plan
        self._plans[target_date] = plan

        logger.info(
            f"Generated Monday plan for {target_date}: "
            f"{len(plan.primary_focus)} primary, "
            f"{len(plan.secondary_focus)} secondary, "
            f"{len(plan.avoid_list)} avoid"
        )

        return plan

    def _build_category_list(
        self,
        category: WatchlistCategory
    ) -> List[WatchlistItem]:
        """Build sorted list for a category."""
        tickers = self._by_category.get(category, [])
        items = []

        for ticker in tickers:
            item = self._candidates.get(ticker)
            if not item:
                continue

            # Apply filters
            if item.overall_score < self.config.min_score_include:
                continue
            if item.risk_score > self.config.max_risk_score:
                continue
            if item.current_price < self.config.min_price:
                continue
            if item.current_price > self.config.max_price:
                continue

            items.append(item)

        # Sort by score
        items.sort(key=lambda x: x.overall_score, reverse=True)

        # Limit per category
        return items[:self.config.max_per_category]

    def _get_next_trading_day(self) -> date:
        """Get next trading day."""
        today = date.today()
        day = today + timedelta(days=1)

        # Skip weekends
        while day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            day += timedelta(days=1)

        return day

    def get_candidate(self, ticker: str) -> Optional[WatchlistItem]:
        """Get a candidate by ticker."""
        return self._candidates.get(ticker.upper())

    def get_candidates_by_category(
        self,
        category: WatchlistCategory
    ) -> List[WatchlistItem]:
        """Get all candidates in a category."""
        tickers = self._by_category.get(category, [])
        return [
            self._candidates[t] for t in tickers
            if t in self._candidates
        ]

    def get_current_plan(self) -> Optional[MondayPlan]:
        """Get current plan."""
        return self._current_plan

    def get_plan(self, target_date: date) -> Optional[MondayPlan]:
        """Get plan for a specific date."""
        return self._plans.get(target_date)

    def update_candidate(
        self,
        ticker: str,
        **updates
    ) -> Optional[WatchlistItem]:
        """Update a candidate's fields."""
        item = self._candidates.get(ticker.upper())
        if item:
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
        return item

    def remove_candidate(self, ticker: str) -> bool:
        """Remove a candidate."""
        ticker = ticker.upper()
        if ticker in self._candidates:
            item = self._candidates.pop(ticker)
            if ticker in self._by_category[item.category]:
                self._by_category[item.category].remove(ticker)
            return True
        return False

    def clear(self) -> None:
        """Clear all candidates."""
        self._candidates.clear()
        for cat in self._by_category:
            self._by_category[cat].clear()

    def export_plan_summary(self, plan: Optional[MondayPlan] = None) -> str:
        """Export plan as text summary."""
        plan = plan or self._current_plan
        if not plan:
            return "No plan generated"

        lines = [
            f"=== MONDAY TRADING PLAN ===",
            f"Target Date: {plan.target_date}",
            f"Market Bias: {plan.market_bias}",
            f"Generated: {plan.created_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"PRIMARY FOCUS ({len(plan.primary_focus)}):",
        ]

        for ticker in plan.primary_focus:
            item = self._candidates.get(ticker)
            if item:
                lines.append(
                    f"  {ticker}: Score={item.overall_score:.0f} "
                    f"Entry=${item.entry_price or item.current_price:.2f} "
                    f"Target=${item.target_price or 0:.2f}"
                )

        lines.append("")
        lines.append(f"SECONDARY FOCUS ({len(plan.secondary_focus)}):")
        lines.append(f"  {', '.join(plan.secondary_focus[:15])}")

        if plan.avoid_list:
            lines.append("")
            lines.append(f"AVOID ({len(plan.avoid_list)}):")
            lines.append(f"  {', '.join(plan.avoid_list[:10])}")

        lines.append("")
        lines.append(f"Max New Positions: {plan.max_new_positions}")
        lines.append(f"Suggested Cash: {plan.suggested_cash_pct}%")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get prep statistics."""
        return {
            "total_candidates": len(self._candidates),
            "by_category": {
                cat.value: len(tickers)
                for cat, tickers in self._by_category.items()
            },
            "plans_generated": len(self._plans),
            "current_plan_date": (
                self._current_plan.target_date.isoformat()
                if self._current_plan else None
            ),
        }


# Singleton instance
_prep: Optional[MondayPrep] = None


def get_monday_prep(config: Optional[MondayPrepConfig] = None) -> MondayPrep:
    """Get singleton MondayPrep instance."""
    global _prep
    if _prep is None:
        _prep = MondayPrep(config)
    return _prep
