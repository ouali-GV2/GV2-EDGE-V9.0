"""
REQUEST ROUTER V7.0
===================

Routage intelligent des requêtes API.

Responsabilités:
- Sélection de la meilleure clé disponible
- Routing par priorité de tâche
- Fallback automatique
- Reservation de quota pour tâches critiques

Architecture:
- Priority-based routing
- Role-based key selection
- Automatic fallback chain
- Quota reservation for critical tasks
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

from .key_registry import KeyRegistry, APIKeyConfig, TaskRole
from .usage_tracker import UsageTracker, KeyMetrics

logger = get_logger("REQUEST_ROUTER")


# ============================================================================
# Configuration
# ============================================================================

# Quota reservation for critical tasks (percentage of quota)
CRITICAL_QUOTA_RESERVE = 0.30  # 30% reserved for critical

# Priority levels
class Priority(Enum):
    """Request priority levels"""
    CRITICAL = 1      # Pre-halt, execution-blocking checks
    HIGH = 2          # Hot tickers, breaking news
    STANDARD = 3      # Normal operations
    LOW = 4           # Batch, background tasks
    BATCH = 5         # Off-hours batch processing


# Priority to role mapping
PRIORITY_ROLES = {
    Priority.CRITICAL: [TaskRole.CRITICAL.value, TaskRole.PRE_HALT.value, "ALL"],
    Priority.HIGH: [TaskRole.HOT_TICKERS.value, TaskRole.COMPANY_NEWS.value, "ALL"],
    Priority.STANDARD: [TaskRole.COMPANY_NEWS.value, TaskRole.GLOBAL_NEWS.value, "ALL"],
    Priority.LOW: [TaskRole.GLOBAL_NEWS.value, TaskRole.BATCH.value, "ALL"],
    Priority.BATCH: [TaskRole.BATCH.value, TaskRole.BACKUP.value, "ALL"],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RoutingResult:
    """Result of a routing decision"""
    success: bool
    key: Optional[APIKeyConfig] = None
    key_id: Optional[str] = None

    # If failed
    reason: Optional[str] = None
    retry_after_seconds: Optional[int] = None

    # Metadata
    alternatives_checked: int = 0
    selected_from: int = 0


@dataclass
class RoutingStats:
    """Routing statistics"""
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    fallbacks_used: int = 0

    by_priority: Dict[str, int] = None
    by_provider: Dict[str, int] = None

    def __post_init__(self):
        if self.by_priority is None:
            self.by_priority = {}
        if self.by_provider is None:
            self.by_provider = {}


# ============================================================================
# Request Router
# ============================================================================

class RequestRouter:
    """
    Intelligent API request router

    Usage:
        router = RequestRouter(registry, tracker)

        # Route a request
        result = router.route(
            provider="finnhub",
            task_type="COMPANY_NEWS",
            priority=Priority.HIGH
        )

        if result.success:
            # Use result.key.key for API call
            response = api_call(result.key.key)
        else:
            # Handle failure
            logger.warning(f"No key available: {result.reason}")
    """

    def __init__(self, registry: KeyRegistry, tracker: UsageTracker):
        self.registry = registry
        self.tracker = tracker
        self.stats = RoutingStats()

    def route(
        self,
        provider: str,
        task_type: str = None,
        priority: Priority = Priority.STANDARD,
        min_quota: int = 1
    ) -> RoutingResult:
        """
        Route a request to the best available key

        Args:
            provider: API provider (finnhub, grok, etc.)
            task_type: Type of task (COMPANY_NEWS, NLP_CLASSIFY, etc.)
            priority: Request priority level
            min_quota: Minimum quota required

        Returns:
            RoutingResult with selected key or failure reason
        """
        self.stats.total_requests += 1
        self.stats.by_provider[provider] = self.stats.by_provider.get(provider, 0) + 1
        self.stats.by_priority[priority.name] = self.stats.by_priority.get(priority.name, 0) + 1

        # Step 1: Get candidate keys
        candidates = self._get_candidates(provider, task_type, priority)

        if not candidates:
            self.stats.failed_routes += 1
            return RoutingResult(
                success=False,
                reason=f"No keys configured for {provider}/{task_type}",
                alternatives_checked=0
            )

        # Step 2: Filter by availability and quota
        available = self._filter_available(candidates, priority, min_quota)

        if not available:
            self.stats.failed_routes += 1

            # Check if keys exist but are in cooldown
            cooldown_keys = [k for k in candidates if not k.is_available()]
            if cooldown_keys:
                # Find shortest cooldown
                min_cooldown = min(
                    (k.cooldown_until - datetime.utcnow()).total_seconds()
                    for k in cooldown_keys if k.cooldown_until
                )
                return RoutingResult(
                    success=False,
                    reason="All keys in cooldown",
                    retry_after_seconds=max(1, int(min_cooldown)),
                    alternatives_checked=len(candidates)
                )

            return RoutingResult(
                success=False,
                reason="All keys exhausted (quota limit)",
                alternatives_checked=len(candidates)
            )

        # Step 3: Select best key
        best_key = self._select_best(available, priority)

        self.stats.successful_routes += 1

        return RoutingResult(
            success=True,
            key=best_key,
            key_id=best_key.id,
            alternatives_checked=len(candidates),
            selected_from=len(available)
        )

    def _get_candidates(
        self,
        provider: str,
        task_type: str,
        priority: Priority
    ) -> List[APIKeyConfig]:
        """Get candidate keys for a request"""

        # Get all keys for provider
        all_keys = self.registry.get_keys(provider)

        if not all_keys:
            return []

        # If task_type specified, filter by role
        if task_type:
            role_keys = [k for k in all_keys if k.has_role(task_type)]
            if role_keys:
                return role_keys

        # Fallback: use priority-based role mapping
        priority_roles = PRIORITY_ROLES.get(priority, ["ALL"])

        for role in priority_roles:
            role_keys = [k for k in all_keys if k.has_role(role)]
            if role_keys:
                return role_keys

        # Last resort: return all keys
        return all_keys

    def _filter_available(
        self,
        keys: List[APIKeyConfig],
        priority: Priority,
        min_quota: int
    ) -> List[APIKeyConfig]:
        """Filter keys by availability and quota"""

        available = []

        for key in keys:
            # Check basic availability
            if not key.is_available():
                continue

            # Get metrics
            metrics = self.tracker.get_metrics(key.id, key.quota_per_minute)

            # Check quota
            effective_quota = self._get_effective_quota(
                metrics.quota_remaining,
                key.quota_per_minute,
                priority
            )

            if effective_quota < min_quota:
                continue

            # Check health (skip very unhealthy keys except for batch)
            if priority != Priority.BATCH and metrics.health_score < 0.3:
                continue

            available.append(key)

        return available

    def _get_effective_quota(
        self,
        remaining: int,
        total: int,
        priority: Priority
    ) -> int:
        """
        Get effective quota considering reservations

        Critical tasks get full quota access.
        Lower priority tasks may not access reserved quota.
        """
        if priority == Priority.CRITICAL:
            return remaining

        # Reserve quota for critical tasks
        reserved = int(total * CRITICAL_QUOTA_RESERVE)
        effective = remaining - reserved

        return max(0, effective)

    def _select_best(
        self,
        keys: List[APIKeyConfig],
        priority: Priority
    ) -> APIKeyConfig:
        """Select the best key from available candidates"""

        # Score each key
        scored = []

        for key in keys:
            metrics = self.tracker.get_metrics(key.id, key.quota_per_minute)
            score = self._calculate_key_score(key, metrics, priority)
            scored.append((key, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Check if using fallback (not primary choice)
        if len(scored) > 1 and scored[0][0].priority > 1:
            self.stats.fallbacks_used += 1

        return scored[0][0]

    def _calculate_key_score(
        self,
        key: APIKeyConfig,
        metrics: KeyMetrics,
        priority: Priority
    ) -> float:
        """Calculate a score for key selection"""
        score = 0.0

        # Base score from key priority (lower = better)
        score += (10 - key.priority) * 10

        # Health score contribution
        score += metrics.health_score * 30

        # Quota remaining contribution
        quota_pct = metrics.quota_remaining / metrics.quota_per_minute if metrics.quota_per_minute > 0 else 0
        score += quota_pct * 25

        # Latency contribution (lower = better)
        if metrics.avg_latency_ms > 0:
            latency_score = max(0, 1 - (metrics.avg_latency_ms / 5000))
            score += latency_score * 15

        # Error rate penalty
        score -= metrics.error_rate * 20

        return score

    def get_fallback_chain(self, provider: str) -> List[str]:
        """Get ordered fallback chain for a provider"""
        keys = self.registry.get_keys(provider)

        # Sort by priority
        keys.sort(key=lambda k: k.priority)

        return [k.id for k in keys]

    def reserve_quota(
        self,
        key_id: str,
        amount: int
    ) -> bool:
        """
        Reserve quota for an upcoming batch of requests

        Note: This is advisory only - actual quota is enforced per-call
        """
        key = self.registry.get_key(key_id)
        if not key:
            return False

        metrics = self.tracker.get_metrics(key_id, key.quota_per_minute)
        return metrics.quota_remaining >= amount

    def get_stats(self) -> Dict:
        """Get routing statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "successful_routes": self.stats.successful_routes,
            "failed_routes": self.stats.failed_routes,
            "success_rate": self.stats.successful_routes / self.stats.total_requests if self.stats.total_requests > 0 else 0,
            "fallbacks_used": self.stats.fallbacks_used,
            "by_priority": self.stats.by_priority,
            "by_provider": self.stats.by_provider
        }

    def reset_stats(self):
        """Reset routing statistics"""
        self.stats = RoutingStats()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_router(registry: KeyRegistry = None, tracker: UsageTracker = None) -> RequestRouter:
    """Create a router with default or provided components"""
    from .key_registry import get_registry
    from .usage_tracker import get_tracker

    if registry is None:
        registry = get_registry()
    if tracker is None:
        tracker = get_tracker()

    return RequestRouter(registry, tracker)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "RequestRouter",
    "RoutingResult",
    "RoutingStats",
    "Priority",
    "create_router",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from .key_registry import KeyRegistry, APIKeyConfig
    from .usage_tracker import UsageTracker

    print("=" * 60)
    print("REQUEST ROUTER TEST")
    print("=" * 60)

    # Setup
    registry = KeyRegistry(db_path="data/test_router_keys.db")
    tracker = UsageTracker()

    # Register test keys
    registry.register_key(APIKeyConfig(
        id="FH_A",
        provider="finnhub",
        key="test_key_a",
        tier="free",
        roles=["HOT_TICKERS", "COMPANY_NEWS"],
        priority=1
    ))

    registry.register_key(APIKeyConfig(
        id="FH_B",
        provider="finnhub",
        key="test_key_b",
        tier="free",
        roles=["GLOBAL_NEWS", "BATCH"],
        priority=2
    ))

    registry.register_key(APIKeyConfig(
        id="FH_BACKUP",
        provider="finnhub",
        key="test_key_backup",
        tier="free",
        roles=["BACKUP", "ALL"],
        priority=3
    ))

    # Set quotas
    for key_id in ["FH_A", "FH_B", "FH_BACKUP"]:
        tracker.set_quota(key_id, 60)

    # Create router
    router = RequestRouter(registry, tracker)

    # Test routing
    print("\n--- Test 1: Route to HOT_TICKERS ---")
    result = router.route("finnhub", "HOT_TICKERS", Priority.HIGH)
    print(f"  Success: {result.success}")
    print(f"  Key: {result.key_id}")
    print(f"  Checked: {result.alternatives_checked}, Selected from: {result.selected_from}")

    print("\n--- Test 2: Route to BATCH ---")
    result = router.route("finnhub", "BATCH", Priority.BATCH)
    print(f"  Success: {result.success}")
    print(f"  Key: {result.key_id}")

    print("\n--- Test 3: Route after exhausting FH_A ---")
    # Simulate heavy usage on FH_A
    for _ in range(60):
        tracker.record_call("FH_A", 100, True)

    result = router.route("finnhub", "HOT_TICKERS", Priority.HIGH)
    print(f"  Success: {result.success}")
    print(f"  Key: {result.key_id}")  # Should fallback to FH_B or BACKUP

    print("\n--- Routing Stats ---")
    print(router.get_stats())

    print("\n--- Fallback Chain ---")
    print(router.get_fallback_chain("finnhub"))
