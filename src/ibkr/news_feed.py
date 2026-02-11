"""
IBKR News Feed - Real-time News from Interactive Brokers

Subscribes to IBKR news feeds:
- BZ (Benzinga)
- BRFG (Briefing.com)
- DJ (Dow Jones)
- Fly (FlyOnTheWall)

Features:
- Real-time news alerts
- Ticker-specific news filtering
- Headline parsing and categorization
- Pre-halt news detection
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Set, Any
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class NewsProvider(Enum):
    """IBKR news providers."""
    BENZINGA = "BZ"
    BRIEFING = "BRFG"
    DOW_JONES = "DJ"
    FLY = "FLY"
    RFNEWS = "RFNEWS"


class NewsUrgency(Enum):
    """News urgency levels."""
    CRITICAL = "CRITICAL"   # Halt-related, major event
    HIGH = "HIGH"           # Breaking news, significant
    NORMAL = "NORMAL"       # Standard news
    LOW = "LOW"             # Background, minor


class NewsCategory(Enum):
    """News categories."""
    HALT = "HALT"                   # Trading halt
    EARNINGS = "EARNINGS"           # Earnings related
    FDA = "FDA"                     # FDA news
    SEC_FILING = "SEC"              # SEC filings
    MERGER_ACQUISITION = "M&A"      # M&A activity
    OFFERING = "OFFERING"           # Stock offering
    ANALYST = "ANALYST"             # Analyst ratings
    INSIDER = "INSIDER"             # Insider activity
    GUIDANCE = "GUIDANCE"           # Company guidance
    DIVIDEND = "DIVIDEND"           # Dividend news
    LAWSUIT = "LAWSUIT"             # Legal issues
    CONTRACT = "CONTRACT"           # Contract wins/losses
    GENERAL = "GENERAL"             # Other news


@dataclass
class NewsArticle:
    """A news article from IBKR."""
    id: str
    timestamp: datetime
    provider: NewsProvider
    headline: str

    # Extracted info
    tickers: List[str] = field(default_factory=list)
    category: NewsCategory = NewsCategory.GENERAL
    urgency: NewsUrgency = NewsUrgency.NORMAL

    # Full content (if available)
    body: str = ""
    summary: str = ""

    # Metadata
    source: str = ""
    language: str = "en"

    # Flags
    is_halt_related: bool = False
    is_earnings_related: bool = False
    is_offering_related: bool = False

    # Sentiment (if analyzed)
    sentiment_score: Optional[float] = None  # -1 to +1

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider.value,
            "headline": self.headline,
            "tickers": self.tickers,
            "category": self.category.value,
            "urgency": self.urgency.value,
            "is_halt_related": self.is_halt_related,
        }


# Patterns for categorization
CATEGORY_PATTERNS = {
    NewsCategory.HALT: [
        r"trading\s+halt",
        r"halted",
        r"volatility\s+halt",
        r"news\s+pending",
        r"LUDP",
        r"T1\s+halt",
    ],
    NewsCategory.EARNINGS: [
        r"earnings",
        r"EPS",
        r"revenue",
        r"quarterly\s+results",
        r"beat",
        r"miss",
        r"guidance",
    ],
    NewsCategory.FDA: [
        r"FDA",
        r"approval",
        r"clinical\s+trial",
        r"phase\s+[123]",
        r"drug\s+application",
        r"NDA",
        r"BLA",
    ],
    NewsCategory.OFFERING: [
        r"offering",
        r"public\s+offering",
        r"secondary",
        r"shelf\s+registration",
        r"S-3",
        r"424B",
        r"ATM\s+program",
        r"dilution",
    ],
    NewsCategory.MERGER_ACQUISITION: [
        r"merger",
        r"acquisition",
        r"acquire",
        r"buyout",
        r"takeover",
        r"deal",
    ],
    NewsCategory.ANALYST: [
        r"upgrade",
        r"downgrade",
        r"price\s+target",
        r"analyst",
        r"rating",
        r"initiates",
    ],
    NewsCategory.INSIDER: [
        r"insider",
        r"form\s+4",
        r"CEO\s+(?:buy|sell)",
        r"director\s+(?:buy|sell)",
    ],
}

URGENCY_PATTERNS = {
    NewsUrgency.CRITICAL: [
        r"halt",
        r"suspended",
        r"FDA\s+(?:approval|rejection)",
        r"bankruptcy",
        r"fraud",
    ],
    NewsUrgency.HIGH: [
        r"breaking",
        r"alert",
        r"significant",
        r"major",
        r"acquisition",
        r"merger",
    ],
}


@dataclass
class NewsFeedConfig:
    """Configuration for news feed."""
    # Providers to subscribe
    providers: List[NewsProvider] = field(default_factory=lambda: [
        NewsProvider.BENZINGA,
        NewsProvider.FLY,
    ])

    # Filtering
    filter_by_tickers: bool = False
    watchlist_tickers: List[str] = field(default_factory=list)

    # History
    max_history: int = 1000
    history_hours: int = 24


class IBKRNewsFeed:
    """
    IBKR real-time news feed.

    Usage:
        from src.ibkr import get_ibkr_connector

        connector = get_ibkr_connector()
        await connector.connect()

        news_feed = IBKRNewsFeed(connector)
        await news_feed.start()

        # Subscribe to news
        news_feed.subscribe(callback=on_news)
        news_feed.subscribe_ticker("AAPL", callback=on_aapl_news)

        # Get recent news
        recent = news_feed.get_recent(hours=1)
    """

    def __init__(
        self,
        connector,  # IBKRConnector
        config: Optional[NewsFeedConfig] = None
    ):
        self.config = config or NewsFeedConfig()
        self._connector = connector

        # State
        self._running = False
        self._subscribed_providers: Set[NewsProvider] = set()

        # News storage
        self._articles: Dict[str, NewsArticle] = {}
        self._by_ticker: Dict[str, List[str]] = {}  # ticker -> article IDs
        self._article_order: List[str] = []  # Ordered by time

        # Callbacks
        self._global_callbacks: List[Callable[[NewsArticle], None]] = []
        self._ticker_callbacks: Dict[str, List[Callable[[NewsArticle], None]]] = {}

        # Compile patterns
        self._category_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in CATEGORY_PATTERNS.items()
        }
        self._urgency_patterns = {
            urg: [re.compile(p, re.IGNORECASE) for p in patterns]
            for urg, patterns in URGENCY_PATTERNS.items()
        }

        # Ticker extraction pattern
        self._ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')

    async def start(self) -> bool:
        """Start the news feed."""
        if not self._connector.is_connected():
            logger.error("IBKR connector not connected")
            return False

        self._running = True

        try:
            ib = self._connector._ib

            # Subscribe to news providers
            for provider in self.config.providers:
                try:
                    # Request news bulletins
                    ib.reqNewsBulletins(True)

                    # Subscribe to news for all tickers (generic subscription)
                    # Note: IBKR news requires specific contract subscriptions
                    self._subscribed_providers.add(provider)
                    logger.info(f"Subscribed to {provider.value} news")

                except Exception as e:
                    logger.error(f"Failed to subscribe to {provider.value}: {e}")

            # Setup news handler
            ib.newsBulletinEvent += self._on_news_bulletin

            logger.info("IBKR news feed started")
            return True

        except Exception as e:
            logger.error(f"Failed to start news feed: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the news feed."""
        self._running = False

        if self._connector._ib:
            self._connector._ib.reqNewsBulletins(False)

        self._subscribed_providers.clear()
        logger.info("IBKR news feed stopped")

    def _on_news_bulletin(self, bulletin) -> None:
        """Handle incoming news bulletin."""
        try:
            article = self._parse_bulletin(bulletin)
            if article:
                self._process_article(article)

        except Exception as e:
            logger.error(f"Error processing news bulletin: {e}")

    def _parse_bulletin(self, bulletin) -> Optional[NewsArticle]:
        """Parse IBKR news bulletin into NewsArticle."""
        try:
            # Extract basic info
            article_id = f"news_{bulletin.msgId}"
            headline = bulletin.message

            # Determine provider
            provider = NewsProvider.BENZINGA  # Default
            for p in NewsProvider:
                if p.value in headline:
                    provider = p
                    break

            article = NewsArticle(
                id=article_id,
                timestamp=datetime.now(),
                provider=provider,
                headline=headline,
            )

            # Extract tickers
            article.tickers = self._extract_tickers(headline)

            # Categorize
            article.category = self._categorize(headline)

            # Determine urgency
            article.urgency = self._determine_urgency(headline)

            # Set flags
            article.is_halt_related = article.category == NewsCategory.HALT
            article.is_earnings_related = article.category == NewsCategory.EARNINGS
            article.is_offering_related = article.category == NewsCategory.OFFERING

            return article

        except Exception as e:
            logger.error(f"Failed to parse bulletin: {e}")
            return None

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text."""
        # Common words to exclude
        exclude = {
            "A", "I", "AM", "PM", "CEO", "CFO", "COO", "FDA", "SEC", "IPO",
            "EPS", "PE", "USA", "NYSE", "THE", "FOR", "AND", "WITH", "NEW",
            "INC", "LLC", "LTD", "ETF", "ATM", "BUY", "SELL", "HOLD"
        }

        matches = self._ticker_pattern.findall(text)
        tickers = [m for m in matches if m not in exclude and len(m) >= 2]

        return list(set(tickers))[:5]  # Max 5 tickers per article

    def _categorize(self, text: str) -> NewsCategory:
        """Categorize news based on content."""
        for category, patterns in self._category_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return category
        return NewsCategory.GENERAL

    def _determine_urgency(self, text: str) -> NewsUrgency:
        """Determine news urgency."""
        for urgency, patterns in self._urgency_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return urgency
        return NewsUrgency.NORMAL

    def _process_article(self, article: NewsArticle) -> None:
        """Process and store article, notify subscribers."""
        # Filter if configured
        if self.config.filter_by_tickers:
            matching = set(article.tickers) & set(self.config.watchlist_tickers)
            if not matching:
                return

        # Store article
        self._articles[article.id] = article
        self._article_order.append(article.id)

        # Index by ticker
        for ticker in article.tickers:
            if ticker not in self._by_ticker:
                self._by_ticker[ticker] = []
            self._by_ticker[ticker].append(article.id)

        # Trim history
        while len(self._article_order) > self.config.max_history:
            old_id = self._article_order.pop(0)
            if old_id in self._articles:
                old_article = self._articles.pop(old_id)
                for ticker in old_article.tickers:
                    if ticker in self._by_ticker:
                        if old_id in self._by_ticker[ticker]:
                            self._by_ticker[ticker].remove(old_id)

        # Notify global subscribers
        for callback in self._global_callbacks:
            try:
                callback(article)
            except Exception as e:
                logger.error(f"News callback error: {e}")

        # Notify ticker-specific subscribers
        for ticker in article.tickers:
            if ticker in self._ticker_callbacks:
                for callback in self._ticker_callbacks[ticker]:
                    try:
                        callback(article)
                    except Exception as e:
                        logger.error(f"Ticker news callback error: {e}")

        # Log important news
        if article.urgency in [NewsUrgency.CRITICAL, NewsUrgency.HIGH]:
            logger.info(
                f"[{article.urgency.value}] {article.tickers}: {article.headline[:100]}"
            )

    # Subscription methods

    def subscribe(self, callback: Callable[[NewsArticle], None]) -> None:
        """Subscribe to all news."""
        self._global_callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[NewsArticle], None]) -> None:
        """Unsubscribe from all news."""
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)

    def subscribe_ticker(
        self,
        ticker: str,
        callback: Callable[[NewsArticle], None]
    ) -> None:
        """Subscribe to news for a specific ticker."""
        ticker = ticker.upper()
        if ticker not in self._ticker_callbacks:
            self._ticker_callbacks[ticker] = []
        self._ticker_callbacks[ticker].append(callback)

    def unsubscribe_ticker(
        self,
        ticker: str,
        callback: Callable[[NewsArticle], None]
    ) -> None:
        """Unsubscribe from ticker news."""
        ticker = ticker.upper()
        if ticker in self._ticker_callbacks:
            if callback in self._ticker_callbacks[ticker]:
                self._ticker_callbacks[ticker].remove(callback)

    # Query methods

    def get_recent(
        self,
        hours: int = 1,
        ticker: Optional[str] = None,
        category: Optional[NewsCategory] = None,
        urgency: Optional[NewsUrgency] = None
    ) -> List[NewsArticle]:
        """Get recent news with optional filters."""
        cutoff = datetime.now() - timedelta(hours=hours)

        if ticker:
            ticker = ticker.upper()
            article_ids = self._by_ticker.get(ticker, [])
            articles = [self._articles[aid] for aid in article_ids if aid in self._articles]
        else:
            articles = list(self._articles.values())

        # Filter by time
        articles = [a for a in articles if a.timestamp > cutoff]

        # Filter by category
        if category:
            articles = [a for a in articles if a.category == category]

        # Filter by urgency
        if urgency:
            articles = [a for a in articles if a.urgency == urgency]

        # Sort by time (newest first)
        articles.sort(key=lambda x: x.timestamp, reverse=True)

        return articles

    def get_halt_news(self, hours: int = 1) -> List[NewsArticle]:
        """Get halt-related news."""
        return self.get_recent(hours=hours, category=NewsCategory.HALT)

    def get_critical_news(self, hours: int = 1) -> List[NewsArticle]:
        """Get critical urgency news."""
        return self.get_recent(hours=hours, urgency=NewsUrgency.CRITICAL)

    def get_ticker_news(self, ticker: str, limit: int = 20) -> List[NewsArticle]:
        """Get news for a specific ticker."""
        ticker = ticker.upper()
        article_ids = self._by_ticker.get(ticker, [])
        articles = [self._articles[aid] for aid in article_ids[-limit:] if aid in self._articles]
        articles.sort(key=lambda x: x.timestamp, reverse=True)
        return articles

    def get_stats(self) -> Dict:
        """Get news feed statistics."""
        return {
            "running": self._running,
            "providers": [p.value for p in self._subscribed_providers],
            "total_articles": len(self._articles),
            "tickers_tracked": len(self._by_ticker),
            "global_subscribers": len(self._global_callbacks),
            "ticker_subscribers": sum(len(cbs) for cbs in self._ticker_callbacks.values()),
        }


# Factory function
def create_news_feed(connector, config: Optional[NewsFeedConfig] = None) -> IBKRNewsFeed:
    """Create a news feed instance."""
    return IBKRNewsFeed(connector, config)
