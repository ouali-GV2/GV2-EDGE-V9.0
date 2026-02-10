"""
GV2-EDGE V7.0 API Pool
======================

Multi-key API management for scalable data ingestion.

Architecture:
- KeyRegistry: Storage and configuration of API keys
- UsageTracker: Real-time usage metrics and rate limiting
- RequestRouter: Intelligent routing by priority and role
- APIPoolManager: Central orchestrator (main entry point)

Usage:
    from src.api_pool import get_pool_manager, Priority

    # Initialize (auto-loads keys from environment)
    pool = get_pool_manager()
    pool.setup()

    # Simple usage
    key_info = pool.get_key("finnhub", "COMPANY_NEWS", Priority.HIGH)
    if key_info.success:
        response = api_call(headers={"X-Finnhub-Token": key_info.key})
        pool.release(key_info.key_id, success=True, latency_ms=120)

    # Context manager (recommended)
    async with pool.acquire("finnhub", "COMPANY_NEWS") as key:
        if key:
            response = await fetch(url, headers={"Token": key})

    # Check status
    status = pool.get_status()
    print(status["keys"])

Supported Providers:
- finnhub: News, quotes, calendars
- grok: NLP classification (xAI)
- reddit: Social buzz (PRAW)
- stocktwits: Social sentiment

Priority Levels:
- CRITICAL: Pre-halt checks, execution-blocking
- HIGH: Hot tickers, breaking news
- STANDARD: Normal operations
- LOW: Background tasks
- BATCH: Off-hours processing
"""

# Key Registry
from .key_registry import (
    KeyRegistry,
    APIKeyConfig,
    KeyStatus,
    TaskRole,
    get_registry,
    setup_default_keys,
)

# Usage Tracker
from .usage_tracker import (
    UsageTracker,
    KeyMetrics,
    CallRecord,
    get_tracker,
)

# Request Router
from .request_router import (
    RequestRouter,
    RoutingResult,
    RoutingStats,
    Priority,
    create_router,
)

# Pool Manager (main entry point)
from .pool_manager import (
    APIPoolManager,
    KeyAcquisition,
    get_pool_manager,
)

__all__ = [
    # Key Registry
    "KeyRegistry",
    "APIKeyConfig",
    "KeyStatus",
    "TaskRole",
    "get_registry",
    "setup_default_keys",

    # Usage Tracker
    "UsageTracker",
    "KeyMetrics",
    "CallRecord",
    "get_tracker",

    # Request Router
    "RequestRouter",
    "RoutingResult",
    "RoutingStats",
    "Priority",
    "create_router",

    # Pool Manager
    "APIPoolManager",
    "KeyAcquisition",
    "get_pool_manager",
]
