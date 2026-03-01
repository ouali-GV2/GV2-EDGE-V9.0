"""
NEWS FLOW SCREENER V6.1 - Global News → Ticker Mapping
=======================================================

Architecture inversee pour detection anticipative:

AVANT (inefficace):
    Pour chaque ticker → chercher ses news → analyser
    Probleme: 500 tickers × API calls = lent + rate limits

MAINTENANT V6.1 (efficace + 100% sources reelles):
    1. Fetch ALL breaking news via V6.1 Ingestors
       - SEC EDGAR 8-K (FREE)
       - Finnhub general news (FREE)
    2. Keyword filter (fast, NO LLM)
    3. Extract tickers mentionnes
    4. NLP classification via Grok (EVENT_TYPE only)
    5. Output: {ticker: [events]} pret pour scoring

IMPORTANT V6.1:
- NO Polygon-via-Grok simulation
- Grok = NLP classification ONLY
- 100% real data sources
"""

import re
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from utils.logger import get_logger
from utils.cache import Cache

# V6.1 Ingestors (real sources)
from src.ingestors.global_news_ingestor import GlobalNewsIngestor, get_global_ingestor
from src.ingestors.sec_filings_ingestor import SECIngestor

# V6.1 Processors
from src.processors.keyword_filter import get_keyword_filter, FilterPriority
from src.processors.ticker_extractor import get_ticker_extractor
from src.processors.nlp_classifier import get_classifier, classify_news

from config import GROK_API_KEY, FINNHUB_API_KEY

logger = get_logger("NEWS_FLOW_SCREENER_V6")

# Cache pour eviter doublons
news_cache = Cache(ttl=900)  # 15 min


# ============================
# EVENT TYPES - UNIFIED TAXONOMY V6
# ============================

EVENT_TYPES = {
    # TIER 1 - CRITICAL (0.90-1.00)
    'FDA_APPROVAL': {'base_impact': 0.95},
    'PDUFA_DECISION': {'base_impact': 0.92},
    'BUYOUT_CONFIRMED': {'base_impact': 0.90},

    # TIER 2 - HIGH (0.75-0.89)
    'FDA_TRIAL_POSITIVE': {'base_impact': 0.85},
    'BREAKTHROUGH_DESIGNATION': {'base_impact': 0.82},
    'FDA_FAST_TRACK': {'base_impact': 0.80},
    'MERGER_ACQUISITION': {'base_impact': 0.85},
    'EARNINGS_BEAT_BIG': {'base_impact': 0.82},
    'MAJOR_CONTRACT': {'base_impact': 0.78},

    # TIER 3 - MEDIUM-HIGH (0.60-0.74)
    'GUIDANCE_RAISE': {'base_impact': 0.70},
    'EARNINGS_BEAT': {'base_impact': 0.65},
    'PARTNERSHIP': {'base_impact': 0.62},
    'PRICE_TARGET_RAISE': {'base_impact': 0.60},

    # TIER 4 - MEDIUM (0.45-0.59)
    'ANALYST_UPGRADE': {'base_impact': 0.52},
    'SHORT_SQUEEZE_SIGNAL': {'base_impact': 0.55},
    'UNUSUAL_VOLUME_NEWS': {'base_impact': 0.48},

    # TIER 5 - SPECULATIVE (0.30-0.44)
    'BUYOUT_RUMOR': {'base_impact': 0.42},
    'SOCIAL_MEDIA_SURGE': {'base_impact': 0.38},
    'BREAKING_POSITIVE': {'base_impact': 0.35},

    # Fallback
    'NONE': {'base_impact': 0.0}
}


# ============================
# DATA CLASSES
# ============================

@dataclass
class NewsEvent:
    """Single news event with extracted data"""
    headline: str
    summary: str
    tickers: List[str]
    event_type: str
    impact_score: float
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    source: str
    published_at: str
    url: Optional[str] = None
    tier: int = 5


@dataclass
class TickerEvents:
    """All events for a single ticker"""
    ticker: str
    events: List[NewsEvent]
    total_impact: float
    dominant_event_type: str
    event_count: int


# ============================
# V6.1 NEWS FETCHING (REAL SOURCES ONLY)
# ============================

async def fetch_news_v6(universe_tickers: Set[str], hours_back: int = 6) -> List[Dict]:
    """
    Fetch news from V6.1 real sources

    Sources:
    - SEC EDGAR 8-K (FREE, critical priority)
    - Finnhub General News (FREE)

    NO Polygon-via-Grok simulation!
    """
    logger.info(f"Fetching news V6.1 (last {hours_back}h)...")

    all_items = []

    # Use V6.1 Global News Ingestor
    ingestor = get_global_ingestor(universe_tickers)
    result = await ingestor.scan(hours_back=hours_back)

    # Convert to standard format
    for item in result.news_items:
        all_items.append({
            'id': item.id,
            'title': item.headline,
            'description': item.summary,
            'tickers': item.tickers,
            'published': item.published_at.isoformat(),
            'url': item.url,
            'source': item.source,
            'source_priority': item.source_priority,
            'filter_priority': item.filter_priority.name if item.filter_priority else 'LOW',
            'filter_category': item.filter_category
        })

    logger.info(f"V6.1: fetched {len(all_items)} news items")

    return all_items


def fetch_news_sync(universe_tickers: Set[str], hours_back: int = 6) -> List[Dict]:
    """
    Synchronous wrapper for fetch_news_v6
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(fetch_news_v6(universe_tickers, hours_back))


# ============================
# NLP PROCESSING (Grok for classification ONLY)
# ============================

async def classify_events_v6(news_items: List[Dict], universe_tickers: Set[str]) -> List[NewsEvent]:
    """
    Classify news items using V6.1 NLP Classifier

    Grok is used ONLY for EVENT_TYPE classification.
    NO data sourcing via Grok!
    """
    if not news_items:
        return []

    logger.info(f"Classifying {len(news_items)} news items...")

    classifier = get_classifier()
    ticker_extractor = get_ticker_extractor(universe_tickers)
    events = []

    for item in news_items:
        headline = item.get('title', '')
        summary = item.get('description', '')

        # Skip if no content
        if not headline:
            continue

        # Extract tickers
        existing_tickers = item.get('tickers', [])
        extracted = ticker_extractor.extract_validated(f"{headline} {summary}")
        all_tickers = list(set(existing_tickers + extracted))

        # Filter to universe
        valid_tickers = [t for t in all_tickers if t in universe_tickers]

        if not valid_tickers:
            continue

        # NLP Classification (Grok)
        try:
            result = await classifier.classify(headline, summary)

            # Skip low impact
            if result.impact < 0.3 or result.event_type == "NONE":
                continue

            event = NewsEvent(
                headline=headline,
                summary=summary[:200] if summary else "",
                tickers=valid_tickers,
                event_type=result.event_type,
                impact_score=result.impact,
                sentiment="BULLISH" if result.impact >= 0.5 else "NEUTRAL",
                source=item.get('source', 'unknown'),
                published_at=item.get('published', datetime.now(timezone.utc).isoformat()),
                url=item.get('url'),
                tier=result.tier
            )
            events.append(event)

        except Exception as e:
            logger.warning(f"Classification error: {e}")
            continue

    logger.info(f"Classified {len(events)} valid events")
    return events


def classify_events_sync(news_items: List[Dict], universe_tickers: Set[str]) -> List[NewsEvent]:
    """Synchronous wrapper"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(classify_events_v6(news_items, universe_tickers))


# ============================
# AGGREGATION
# ============================

def aggregate_events_by_ticker(events: List[NewsEvent]) -> Dict[str, TickerEvents]:
    """
    Aggregate all events by ticker

    Returns dict: ticker → TickerEvents
    """
    ticker_map = defaultdict(list)

    for event in events:
        for ticker in event.tickers:
            ticker_map[ticker].append(event)

    results = {}

    for ticker, ticker_events in ticker_map.items():
        # Calculate total impact (max, not sum)
        impacts = [e.impact_score for e in ticker_events]
        total_impact = max(impacts) if impacts else 0

        # Confluence bonus for multiple events
        if len(ticker_events) > 1:
            total_impact = min(1.0, total_impact + 0.1 * (len(ticker_events) - 1))

        # Find dominant event type (highest impact)
        dominant_event = max(ticker_events, key=lambda e: e.impact_score)

        results[ticker] = TickerEvents(
            ticker=ticker,
            events=ticker_events,
            total_impact=round(total_impact, 3),
            dominant_event_type=dominant_event.event_type,
            event_count=len(ticker_events)
        )

    return results


# ============================
# MAIN SCREENER FUNCTION
# ============================

def run_news_flow_screener(universe_tickers: List[str], hours_back: int = 6) -> Dict[str, TickerEvents]:
    """
    Main entry point for news flow screening (V6.1)

    Flow:
    1. Fetch global news via V6.1 Ingestors (SEC + Finnhub)
    2. Extract and validate tickers
    3. NLP classification with Grok (EVENT_TYPE only)
    4. Aggregate by ticker

    Args:
        universe_tickers: List of valid small cap tickers
        hours_back: How many hours to look back

    Returns:
        Dict[ticker] -> TickerEvents with all relevant events
    """
    logger.info("=" * 60)
    logger.info(f"NEWS FLOW SCREENER V6.1 - Last {hours_back}h")
    logger.info(f"Universe: {len(universe_tickers)} tickers")
    logger.info("Sources: SEC EDGAR + Finnhub (100% real)")
    logger.info("=" * 60)

    universe_set = set(t.upper() for t in universe_tickers)

    # Step 1: Fetch news from V6.1 real sources
    all_news = fetch_news_sync(universe_set, hours_back)

    logger.info(f"Total news fetched: {len(all_news)}")

    if not all_news:
        logger.warning("No news fetched")
        return {}

    # Step 2: NLP Classification (Grok for EVENT_TYPE only)
    events = classify_events_sync(all_news, universe_set)

    if not events:
        logger.info("No valid events extracted")
        return {}

    # Step 3: Aggregate by ticker
    ticker_events = aggregate_events_by_ticker(events)

    # Step 4: Sort by impact
    sorted_tickers = sorted(
        ticker_events.keys(),
        key=lambda t: ticker_events[t].total_impact,
        reverse=True
    )

    # Log results
    logger.info(f"\nNEWS FLOW RESULTS:")
    logger.info("=" * 50)

    for ticker in sorted_tickers[:20]:
        te = ticker_events[ticker]
        logger.info(
            f"  {ticker}: impact={te.total_impact:.2f}, "
            f"type={te.dominant_event_type}, "
            f"events={te.event_count}"
        )

    return ticker_events


# ============================
# ASYNC VERSION
# ============================

async def run_news_flow_screener_async(
    universe_tickers: List[str],
    hours_back: int = 6
) -> Dict[str, TickerEvents]:
    """
    Async version of news flow screener
    """
    logger.info("=" * 60)
    logger.info(f"NEWS FLOW SCREENER V6.1 ASYNC - Last {hours_back}h")
    logger.info("=" * 60)

    universe_set = set(t.upper() for t in universe_tickers)

    # Fetch
    all_news = await fetch_news_v6(universe_set, hours_back)

    if not all_news:
        return {}

    # Classify
    events = await classify_events_v6(all_news, universe_set)

    if not events:
        return {}

    # Aggregate
    return aggregate_events_by_ticker(events)


# ============================
# UTILITY FUNCTIONS
# ============================

def get_events_by_type(ticker_events: Dict[str, TickerEvents]) -> Dict[str, List[Dict]]:
    """
    Get tickers grouped by event type
    """
    type_map = defaultdict(list)

    for ticker, te in ticker_events.items():
        type_map[te.dominant_event_type].append({
            'ticker': ticker,
            'impact': te.total_impact,
            'event_count': te.event_count
        })

    # Sort each type by impact
    for event_type in type_map:
        type_map[event_type].sort(key=lambda x: x['impact'], reverse=True)

    return dict(type_map)


def get_calendar_view(ticker_events: Dict[str, TickerEvents]) -> List[Dict]:
    """
    Get chronological view of all events
    """
    all_events = []

    for ticker, te in ticker_events.items():
        for event in te.events:
            all_events.append({
                'ticker': ticker,
                'headline': event.headline,
                'event_type': event.event_type,
                'impact': event.impact_score,
                'tier': event.tier,
                'published_at': event.published_at,
                'sentiment': event.sentiment,
                'source': event.source
            })

    # Sort by time
    all_events.sort(key=lambda x: x['published_at'], reverse=True)

    return all_events


def get_top_catalysts(ticker_events: Dict[str, TickerEvents], limit: int = 10) -> List[Dict]:
    """
    Get top catalysts sorted by impact
    """
    sorted_tickers = sorted(
        ticker_events.items(),
        key=lambda x: x[1].total_impact,
        reverse=True
    )[:limit]

    return [
        {
            'ticker': ticker,
            'impact': te.total_impact,
            'event_type': te.dominant_event_type,
            'event_count': te.event_count,
            'headlines': [e.headline for e in te.events[:3]]
        }
        for ticker, te in sorted_tickers
    ]


# ============================
# MODULE EXPORTS
# ============================

__all__ = [
    "run_news_flow_screener",
    "run_news_flow_screener_async",
    "NewsEvent",
    "TickerEvents",
    "get_events_by_type",
    "get_calendar_view",
    "get_top_catalysts",
]


# ============================
# CLI TEST
# ============================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("NEWS FLOW SCREENER V6.1 - TEST")
    print("100% Real Sources (NO Polygon-via-Grok)")
    print("=" * 60)

    # Test universe
    test_universe = [
        "AAPL", "TSLA", "NVDA", "AMD", "PLTR", "SOFI", "NIO", "LCID",
        "MARA", "RIOT", "COIN", "HOOD", "UPST", "AFRM", "SQ", "PYPL"
    ]

    print(f"\nTest universe: {len(test_universe)} tickers")

    async def test():
        results = await run_news_flow_screener_async(test_universe, hours_back=6)

        print(f"\nResults: {len(results)} tickers with events")

        # Show top catalysts
        top = get_top_catalysts(results, 10)
        print(f"\nTop Catalysts:")
        for cat in top:
            print(f"  {cat['ticker']}: {cat['event_type']} (impact={cat['impact']:.2f})")

        # Show by event type
        by_type = get_events_by_type(results)
        print(f"\nBy Event Type:")
        for event_type, tickers in by_type.items():
            print(f"  {event_type}: {[t['ticker'] for t in tickers[:5]]}")

    asyncio.run(test())
