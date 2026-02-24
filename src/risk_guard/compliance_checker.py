"""
Compliance Checker V8 - Exchange Compliance and Delisting Risk
==============================================================

Monitors and detects:
- NASDAQ/NYSE deficiency notifications
- Minimum bid price violations ($1 rule)
- Minimum equity violations
- Filing delinquency (10-K, 10-Q late)
- Audit opinion issues (going concern)
- Stockholders' equity requirements
- Public float requirements

Risk levels:
- CRITICAL: Delisting imminent, trading halt likely
- HIGH: Active deficiency, grace period running
- MEDIUM: Warning signs, monitoring needed
- LOW: Minor issues, likely to resolve
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class ComplianceIssue(Enum):
    """Types of compliance issues."""
    # Price-based
    BID_PRICE = "BID_PRICE"                  # Below $1 for 30+ days
    CLOSING_PRICE = "CLOSING_PRICE"          # Minimum closing price

    # Financial
    STOCKHOLDERS_EQUITY = "EQUITY"           # Below minimum equity
    MARKET_VALUE = "MARKET_VALUE"            # Below min market value
    NET_INCOME = "NET_INCOME"                # Sustained losses

    # Filing/Disclosure
    FILING_DELINQUENT = "FILING_DELINQUENT"  # Late 10-K/10-Q
    AUDIT_OPINION = "AUDIT_OPINION"          # Going concern, adverse
    DISCLOSURE_ISSUE = "DISCLOSURE"          # Material misstatement

    # Governance
    BOARD_COMPOSITION = "BOARD"              # Independent director issues
    AUDIT_COMMITTEE = "AUDIT_COMMITTEE"      # Committee requirements

    # Trading
    PUBLIC_FLOAT = "PUBLIC_FLOAT"            # Minimum float shares
    SHAREHOLDERS = "SHAREHOLDERS"            # Minimum shareholders
    MARKET_MAKERS = "MARKET_MAKERS"          # Minimum market makers

    # Other
    REVERSE_SPLIT = "REVERSE_SPLIT"          # Pending/announced R/S
    SHELL_COMPANY = "SHELL"                  # Shell company status
    OTHER = "OTHER"


class ComplianceStatus(Enum):
    """Status of compliance issue."""
    NOTICE_RECEIVED = "NOTICE"         # Deficiency notice received
    GRACE_PERIOD = "GRACE"             # In grace/cure period
    PLAN_SUBMITTED = "PLAN"            # Compliance plan submitted
    HEARING_SCHEDULED = "HEARING"      # Panel hearing scheduled
    APPEAL_PENDING = "APPEAL"          # Appeal in progress
    REGAINED = "REGAINED"              # Compliance regained
    DELISTING_PENDING = "DELISTING"    # Delisting determination made


class ComplianceRisk(Enum):
    """Risk severity levels."""
    CRITICAL = "CRITICAL"   # Delisting imminent
    HIGH = "HIGH"           # Active deficiency
    MEDIUM = "MEDIUM"       # Warning signs
    LOW = "LOW"             # Minor issues
    NONE = "NONE"           # Compliant


@dataclass
class ComplianceEvent:
    """Represents a compliance event or notification."""
    ticker: str
    issue_type: ComplianceIssue
    status: ComplianceStatus
    notice_date: datetime

    # Details
    description: str = ""
    exchange: str = ""  # NASDAQ, NYSE, etc.

    # Deadlines
    cure_deadline: Optional[datetime] = None
    hearing_date: Optional[datetime] = None

    # Thresholds
    required_value: Optional[float] = None  # e.g., $1.00 bid price
    current_value: Optional[float] = None   # e.g., $0.45 current bid

    # Resolution
    is_resolved: bool = False
    resolution_date: Optional[datetime] = None

    # Source
    filing_reference: Optional[str] = None  # 8-K accession number

    def days_until_deadline(self) -> Optional[int]:
        """Days until cure deadline."""
        if self.cure_deadline:
            return (self.cure_deadline - datetime.now()).days
        return None

    def is_deadline_imminent(self, days: int = 30) -> bool:
        """Check if deadline is within N days."""
        remaining = self.days_until_deadline()
        return remaining is not None and remaining <= days


@dataclass
class ComplianceProfile:
    """Complete compliance profile for a ticker."""
    ticker: str
    exchange: str = ""
    risk_level: ComplianceRisk = ComplianceRisk.NONE
    risk_score: float = 0.0  # 0-100

    # Active issues
    issues: List[ComplianceEvent] = field(default_factory=list)

    # Current status
    is_compliant: bool = True
    has_active_deficiency: bool = False
    has_delisting_risk: bool = False
    has_pending_reverse_split: bool = False

    # Price tracking (for bid price rule)
    current_bid: Optional[float] = None
    days_below_dollar: int = 0
    consecutive_closes_below: int = 0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    last_notice_date: Optional[datetime] = None

    def get_block_reason(self) -> Optional[str]:
        """Get reason string if trade should be blocked."""
        if self.risk_level == ComplianceRisk.CRITICAL:
            if self.has_delisting_risk:
                return "DELISTING_RISK"
            return "COMPLIANCE_CRITICAL"
        return None

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on risk."""
        multipliers = {
            ComplianceRisk.CRITICAL: 0.25,  # V8: plancher 0.25 — hard block géré par unified_guard si DELISTING_RISK
            ComplianceRisk.HIGH: 0.25,      # 25% of normal
            ComplianceRisk.MEDIUM: 0.50,    # 50% of normal
            ComplianceRisk.LOW: 0.75,       # 75% of normal
            ComplianceRisk.NONE: 1.0        # Full size
        }
        return multipliers.get(self.risk_level, 1.0)

    def get_nearest_deadline(self) -> Optional[datetime]:
        """Get the nearest compliance deadline."""
        deadlines = [
            issue.cure_deadline
            for issue in self.issues
            if issue.cure_deadline and not issue.is_resolved
        ]
        return min(deadlines) if deadlines else None


# Exchange minimum requirements
EXCHANGE_REQUIREMENTS = {
    "NASDAQ": {
        "bid_price": 1.00,
        "market_value": 35_000_000,      # $35M market value listed securities
        "stockholders_equity": 2_500_000, # $2.5M (Capital Market)
        "public_float": 500_000,          # 500K shares
        "shareholders": 300,              # 300 round lot holders
        "market_makers": 2,
    },
    "NYSE": {
        "bid_price": 1.00,
        "market_value": 15_000_000,       # $15M average market cap
        "stockholders_equity": 6_000_000,  # $6M
        "public_float": 600_000,           # 600K shares
        "shareholders": 400,
    },
    "NYSE_ARCA": {
        "bid_price": 1.00,
        "market_value": 5_000_000,
    },
    "NYSE_AMERICAN": {  # Formerly AMEX
        "bid_price": 0.20,  # Lower threshold
        "stockholders_equity": 2_000_000,
    },
}

# Patterns for detecting compliance issues in text
COMPLIANCE_PATTERNS = {
    ComplianceIssue.BID_PRICE: [
        r"bid\s+price\s+(?:deficiency|requirement|rule)",
        r"minimum\s+(?:\$1\.00|one\s+dollar)\s+(?:bid|closing)",
        r"listing\s+rule\s+5550\(a\)\(2\)",  # NASDAQ bid price rule
        r"below\s+\$1\.00\s+for\s+\d+\s+consecutive",
    ],
    ComplianceIssue.STOCKHOLDERS_EQUITY: [
        r"stockholders?\s+equity\s+(?:deficiency|requirement)",
        r"minimum\s+(?:equity|net\s+tangible\s+assets)",
        r"listing\s+rule\s+5550\(b\)",
    ],
    ComplianceIssue.FILING_DELINQUENT: [
        r"(?:10-K|10-Q|annual\s+report)\s+(?:late|delinquent|not\s+filed)",
        r"(?:failure|failed)\s+to\s+(?:file|submit)\s+(?:timely|periodic)",
        r"(?:SEC|exchange)\s+filing\s+(?:delinquency|deficiency)",
    ],
    ComplianceIssue.AUDIT_OPINION: [
        r"going\s+concern",
        r"(?:qualified|adverse)\s+(?:audit\s+)?opinion",
        r"substantial\s+doubt.*continue\s+as\s+a\s+going\s+concern",
    ],
    ComplianceIssue.REVERSE_SPLIT: [
        r"reverse\s+(?:stock\s+)?split",
        r"share\s+consolidation",
        r"(?:1[- ]for[- ]\d+|one[- ]for[- ]\w+)\s+reverse",
    ],
    ComplianceIssue.MARKET_VALUE: [
        r"market\s+value\s+(?:of\s+)?(?:listed\s+)?(?:securities\s+)?(?:deficiency|requirement)",
        r"minimum\s+market\s+(?:value|cap)",
    ],
    ComplianceIssue.PUBLIC_FLOAT: [
        r"public\s+float\s+(?:deficiency|requirement)",
        r"minimum\s+(?:publicly\s+held\s+)?shares",
    ],
}

# Grace periods by issue type (in days)
GRACE_PERIODS = {
    ComplianceIssue.BID_PRICE: 180,          # 180 days initial
    ComplianceIssue.STOCKHOLDERS_EQUITY: 180,
    ComplianceIssue.MARKET_VALUE: 180,
    ComplianceIssue.FILING_DELINQUENT: 60,   # Shorter for filings
    ComplianceIssue.PUBLIC_FLOAT: 180,
    ComplianceIssue.AUDIT_OPINION: 90,
}


class ComplianceChecker:
    """
    Checks and tracks exchange compliance issues.

    Usage:
        checker = ComplianceChecker()

        # Check single ticker
        profile = await checker.analyze(ticker, price_history, filings)
        if profile.risk_level == ComplianceRisk.CRITICAL:
            block_trade()

        # Track price compliance
        checker.update_price(ticker, current_bid=0.85)
    """

    def __init__(self):
        # Cache of compliance profiles
        self._profiles: Dict[str, ComplianceProfile] = {}

        # Known problem tickers
        self._flagged_tickers: Set[str] = set()

        # Cache TTL
        self._cache_ttl = timedelta(hours=4)

        # Compile patterns
        self._patterns = {
            issue: [re.compile(p, re.IGNORECASE) for p in patterns]
            for issue, patterns in COMPLIANCE_PATTERNS.items()
        }

    async def analyze(
        self,
        ticker: str,
        exchange: str = "NASDAQ",
        sec_filings: Optional[List[Dict]] = None,
        news_items: Optional[List[Dict]] = None,
        current_bid: Optional[float] = None,
        price_history: Optional[List[float]] = None,
        force_refresh: bool = False
    ) -> ComplianceProfile:
        """
        Analyze compliance status for a ticker.

        Args:
            ticker: Stock ticker
            exchange: Exchange (NASDAQ, NYSE, etc.)
            sec_filings: List of 8-K and other filings
            news_items: News items that may contain compliance info
            current_bid: Current bid price
            price_history: Recent closing prices (oldest first)
            force_refresh: Force cache refresh

        Returns:
            ComplianceProfile with risk assessment
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._profiles:
            cached = self._profiles[ticker]
            if datetime.now() - cached.last_updated < self._cache_ttl:
                # Update price if provided
                if current_bid is not None:
                    cached.current_bid = current_bid
                return cached

        # Create new profile
        profile = ComplianceProfile(
            ticker=ticker,
            exchange=exchange,
            current_bid=current_bid
        )

        # Analyze SEC filings for compliance issues
        if sec_filings:
            await self._analyze_filings(profile, sec_filings)

        # Analyze news for compliance mentions
        if news_items:
            await self._analyze_news(profile, news_items)

        # Check price compliance
        if price_history:
            self._check_price_compliance(profile, price_history, exchange)
        elif current_bid is not None:
            self._check_single_price(profile, current_bid, exchange)

        # Calculate overall risk
        self._calculate_risk(profile)

        # Cache result
        self._profiles[ticker] = profile

        return profile

    async def _analyze_filings(
        self,
        profile: ComplianceProfile,
        filings: List[Dict]
    ) -> None:
        """Analyze SEC filings for compliance events."""
        for filing in filings:
            form_type = filing.get("form_type", "")
            description = filing.get("description", "")
            content = filing.get("content", description)  # Full text if available

            # 8-K filings often contain compliance notices
            if form_type in ["8-K", "8-K/A"]:
                events = self._parse_compliance_filing(profile.ticker, filing)
                profile.issues.extend(events)

            # NT filings indicate late filing
            if form_type.startswith("NT"):  # NT 10-K, NT 10-Q
                event = ComplianceEvent(
                    ticker=profile.ticker,
                    issue_type=ComplianceIssue.FILING_DELINQUENT,
                    status=ComplianceStatus.NOTICE_RECEIVED,
                    notice_date=self._parse_date(filing.get("filed_date")),
                    description=f"Late filing notification: {form_type}",
                    exchange=profile.exchange,
                    filing_reference=filing.get("accession_number")
                )
                profile.issues.append(event)
                profile.has_active_deficiency = True

    def _parse_compliance_filing(
        self,
        ticker: str,
        filing: Dict
    ) -> List[ComplianceEvent]:
        """Parse an 8-K filing for compliance events."""
        events = []
        description = filing.get("description", "")
        content = filing.get("content", description)
        text = f"{description} {content}"

        for issue_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    # Determine status from text
                    status = self._determine_status(text)

                    event = ComplianceEvent(
                        ticker=ticker,
                        issue_type=issue_type,
                        status=status,
                        notice_date=self._parse_date(filing.get("filed_date")),
                        description=self._extract_description(text, issue_type),
                        filing_reference=filing.get("accession_number")
                    )

                    # Set cure deadline based on grace period
                    grace_days = GRACE_PERIODS.get(issue_type, 180)
                    event.cure_deadline = event.notice_date + timedelta(days=grace_days)

                    events.append(event)
                    break  # One event per issue type per filing

        return events

    async def _analyze_news(
        self,
        profile: ComplianceProfile,
        news_items: List[Dict]
    ) -> None:
        """Analyze news items for compliance mentions."""
        for item in news_items:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            text = f"{headline} {summary}"

            for issue_type, patterns in self._patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        # Check if we already have this issue
                        existing = [
                            e for e in profile.issues
                            if e.issue_type == issue_type
                        ]
                        if not existing:
                            event = ComplianceEvent(
                                ticker=profile.ticker,
                                issue_type=issue_type,
                                status=ComplianceStatus.NOTICE_RECEIVED,
                                notice_date=self._parse_date(item.get("published")),
                                description=headline[:200],
                                exchange=profile.exchange
                            )
                            profile.issues.append(event)
                        break

    def _check_price_compliance(
        self,
        profile: ComplianceProfile,
        price_history: List[float],
        exchange: str
    ) -> None:
        """Check bid price compliance using price history."""
        requirements = EXCHANGE_REQUIREMENTS.get(exchange, EXCHANGE_REQUIREMENTS["NASDAQ"])
        min_price = requirements.get("bid_price", 1.00)

        # Count consecutive days below minimum
        consecutive_below = 0
        for price in reversed(price_history):
            if price < min_price:
                consecutive_below += 1
            else:
                break

        profile.days_below_dollar = consecutive_below
        profile.consecutive_closes_below = consecutive_below

        if price_history:
            profile.current_bid = price_history[-1]

        # 30 consecutive days triggers deficiency notice
        if consecutive_below >= 30:
            profile.has_active_deficiency = True

            # Check if we already have this issue recorded
            existing_bid_issue = [
                e for e in profile.issues
                if e.issue_type == ComplianceIssue.BID_PRICE and not e.is_resolved
            ]

            if not existing_bid_issue:
                event = ComplianceEvent(
                    ticker=profile.ticker,
                    issue_type=ComplianceIssue.BID_PRICE,
                    status=ComplianceStatus.NOTICE_RECEIVED,
                    notice_date=datetime.now() - timedelta(days=consecutive_below - 30),
                    description=f"Bid price below ${min_price:.2f} for {consecutive_below} days",
                    exchange=exchange,
                    required_value=min_price,
                    current_value=profile.current_bid
                )
                event.cure_deadline = event.notice_date + timedelta(days=180)
                profile.issues.append(event)

    def _check_single_price(
        self,
        profile: ComplianceProfile,
        current_bid: float,
        exchange: str
    ) -> None:
        """Check current price against requirements."""
        requirements = EXCHANGE_REQUIREMENTS.get(exchange, EXCHANGE_REQUIREMENTS["NASDAQ"])
        min_price = requirements.get("bid_price", 1.00)

        if current_bid < min_price:
            # Warning but not deficiency (need 30 days)
            profile.is_compliant = False

    def _determine_status(self, text: str) -> ComplianceStatus:
        """Determine compliance status from text."""
        text_lower = text.lower()

        if "delisting" in text_lower and ("determination" in text_lower or "will be" in text_lower):
            return ComplianceStatus.DELISTING_PENDING
        if "hearing" in text_lower and ("scheduled" in text_lower or "requested" in text_lower):
            return ComplianceStatus.HEARING_SCHEDULED
        if "appeal" in text_lower:
            return ComplianceStatus.APPEAL_PENDING
        if "plan" in text_lower and "submitted" in text_lower:
            return ComplianceStatus.PLAN_SUBMITTED
        if "regained" in text_lower or "compliance" in text_lower and "achieved" in text_lower:
            return ComplianceStatus.REGAINED
        if "grace" in text_lower or "cure period" in text_lower:
            return ComplianceStatus.GRACE_PERIOD

        return ComplianceStatus.NOTICE_RECEIVED

    def _extract_description(self, text: str, issue_type: ComplianceIssue) -> str:
        """Extract relevant description snippet."""
        # Find the sentence containing the issue
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            for pattern in self._patterns.get(issue_type, []):
                if pattern.search(sentence):
                    return sentence.strip()[:300]
        return f"{issue_type.value} deficiency detected"

    def _parse_date(self, date_value) -> datetime:
        """Parse date from various formats."""
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            try:
                return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now()

    def _calculate_risk(self, profile: ComplianceProfile) -> None:
        """Calculate overall risk level and score."""
        score = 0.0

        # Check for delisting status
        delisting_issues = [
            e for e in profile.issues
            if e.status == ComplianceStatus.DELISTING_PENDING and not e.is_resolved
        ]
        if delisting_issues:
            profile.has_delisting_risk = True
            score += 60

        # Check for active deficiencies
        active_issues = [e for e in profile.issues if not e.is_resolved]
        profile.has_active_deficiency = len(active_issues) > 0

        for issue in active_issues:
            # Base scores by issue type
            issue_scores = {
                ComplianceIssue.BID_PRICE: 30,
                ComplianceIssue.STOCKHOLDERS_EQUITY: 35,
                ComplianceIssue.FILING_DELINQUENT: 40,
                ComplianceIssue.AUDIT_OPINION: 45,
                ComplianceIssue.MARKET_VALUE: 25,
                ComplianceIssue.PUBLIC_FLOAT: 20,
                ComplianceIssue.REVERSE_SPLIT: 25,
            }
            base = issue_scores.get(issue.issue_type, 15)

            # Status multipliers
            status_mult = {
                ComplianceStatus.DELISTING_PENDING: 2.0,
                ComplianceStatus.HEARING_SCHEDULED: 1.5,
                ComplianceStatus.APPEAL_PENDING: 1.3,
                ComplianceStatus.NOTICE_RECEIVED: 1.0,
                ComplianceStatus.GRACE_PERIOD: 0.9,
                ComplianceStatus.PLAN_SUBMITTED: 0.7,
            }
            mult = status_mult.get(issue.status, 1.0)

            # Urgency based on deadline
            if issue.is_deadline_imminent(7):
                mult *= 1.5
            elif issue.is_deadline_imminent(30):
                mult *= 1.2

            score += base * mult

        # Check for reverse split (often indicates desperation)
        rs_issues = [e for e in profile.issues if e.issue_type == ComplianceIssue.REVERSE_SPLIT]
        if rs_issues:
            profile.has_pending_reverse_split = True
            score += 15

        # Price-based risk
        if profile.days_below_dollar >= 30:
            score += min(20, profile.days_below_dollar / 5)

        # Cap at 100
        profile.risk_score = min(100.0, score)

        # Determine risk level
        if profile.risk_score >= 70 or profile.has_delisting_risk:
            profile.risk_level = ComplianceRisk.CRITICAL
            profile.is_compliant = False
        elif profile.risk_score >= 45:
            profile.risk_level = ComplianceRisk.HIGH
            profile.is_compliant = False
        elif profile.risk_score >= 25:
            profile.risk_level = ComplianceRisk.MEDIUM
        elif profile.risk_score >= 10:
            profile.risk_level = ComplianceRisk.LOW
        else:
            profile.risk_level = ComplianceRisk.NONE
            profile.is_compliant = True

    async def analyze_batch(
        self,
        tickers: List[str],
        exchange: str = "NASDAQ"
    ) -> Dict[str, ComplianceProfile]:
        """Analyze multiple tickers concurrently."""
        tasks = [self.analyze(ticker, exchange) for ticker in tickers]
        profiles = await asyncio.gather(*tasks)
        return {ticker: profile for ticker, profile in zip(tickers, profiles)}

    def update_price(
        self,
        ticker: str,
        current_bid: float,
        exchange: str = "NASDAQ"
    ) -> Optional[ComplianceProfile]:
        """Update price for cached profile."""
        ticker = ticker.upper()
        if ticker in self._profiles:
            profile = self._profiles[ticker]
            old_bid = profile.current_bid

            profile.current_bid = current_bid
            profile.last_updated = datetime.now()

            # Track consecutive days below $1
            requirements = EXCHANGE_REQUIREMENTS.get(exchange, EXCHANGE_REQUIREMENTS["NASDAQ"])
            min_price = requirements.get("bid_price", 1.00)

            if current_bid < min_price:
                profile.days_below_dollar += 1
            else:
                profile.days_below_dollar = 0

            # Recalculate risk
            self._calculate_risk(profile)

            return profile
        return None

    def flag_ticker(self, ticker: str) -> None:
        """Manually flag a ticker for compliance issues."""
        self._flagged_tickers.add(ticker.upper())

    def get_cached_profile(self, ticker: str) -> Optional[ComplianceProfile]:
        """Get cached profile without refresh."""
        return self._profiles.get(ticker.upper())

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cache for a ticker or all tickers."""
        if ticker:
            self._profiles.pop(ticker.upper(), None)
        else:
            self._profiles.clear()

    def get_at_risk_tickers(self) -> List[str]:
        """Get tickers with HIGH or CRITICAL compliance risk."""
        return [
            ticker for ticker, profile in self._profiles.items()
            if profile.risk_level in [ComplianceRisk.HIGH, ComplianceRisk.CRITICAL]
        ]


# Singleton instance
_checker: Optional[ComplianceChecker] = None


def get_compliance_checker() -> ComplianceChecker:
    """Get singleton ComplianceChecker instance."""
    global _checker
    if _checker is None:
        _checker = ComplianceChecker()
    return _checker
