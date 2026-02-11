"""
IBKR Module - Interactive Brokers Integration for GV2-EDGE

Components:
- IBKRConnector: TWS/Gateway connection management
- IBKRNewsFeed: Real-time news from IBKR providers
- IBKRMarketData: Real-time quotes and historical data
- IBKROrderManager: Advanced order management with brackets

Requirements:
    pip install ib_insync

Usage:
    from src.ibkr import get_ibkr_connector, create_news_feed, create_market_data

    # Connect to TWS/Gateway
    connector = get_ibkr_connector()
    await connector.connect()

    # Real-time news
    news_feed = create_news_feed(connector)
    await news_feed.start()
    news_feed.subscribe(on_news_callback)

    # Market data
    market_data = create_market_data(connector)
    await market_data.subscribe("AAPL", on_tick_callback)

    # Place bracket order
    from src.ibkr import create_order_manager, OrderAction

    manager = create_order_manager(connector)
    bracket = await manager.place_bracket(
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=100,
        entry_price=150.0,
        stop_price=147.0,
        target_price=156.0
    )

Connection ports:
- TWS Paper: 7497
- TWS Live: 7496
- Gateway Paper: 4002
- Gateway Live: 4001
"""

# Connector
from .connector import (
    IBKRConnector,
    ConnectionConfig,
    ConnectionState,
    Quote,
    Position,
    Order,
    OrderType,
    OrderAction,
    OrderStatus,
    AccountSummary,
    get_ibkr_connector,
)

# News Feed
from .news_feed import (
    IBKRNewsFeed,
    NewsFeedConfig,
    NewsArticle,
    NewsProvider,
    NewsUrgency,
    NewsCategory,
    create_news_feed,
)

# Market Data
from .market_data import (
    IBKRMarketData,
    MarketDataConfig,
    TickData,
    Bar,
    BarSize,
    WhatToShow,
    ScannerResult,
    MarketScanner,
    create_market_data,
)

# Order Manager
from .order_manager import (
    IBKROrderManager,
    OrderManagerConfig,
    BracketOrder,
    BracketType,
    RiskParameters,
    PositionSizeMethod,
    create_order_manager,
)

__all__ = [
    # Connector
    "IBKRConnector",
    "ConnectionConfig",
    "ConnectionState",
    "Quote",
    "Position",
    "Order",
    "OrderType",
    "OrderAction",
    "OrderStatus",
    "AccountSummary",
    "get_ibkr_connector",
    # News Feed
    "IBKRNewsFeed",
    "NewsFeedConfig",
    "NewsArticle",
    "NewsProvider",
    "NewsUrgency",
    "NewsCategory",
    "create_news_feed",
    # Market Data
    "IBKRMarketData",
    "MarketDataConfig",
    "TickData",
    "Bar",
    "BarSize",
    "WhatToShow",
    "ScannerResult",
    "MarketScanner",
    "create_market_data",
    # Order Manager
    "IBKROrderManager",
    "OrderManagerConfig",
    "BracketOrder",
    "BracketType",
    "RiskParameters",
    "PositionSizeMethod",
    "create_order_manager",
]
