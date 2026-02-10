"""
Batch Processor - Heavy Computation Tasks

Handles computationally expensive operations during off-hours:
- Historical data backfill
- Model training and validation
- Backtesting strategies
- Data aggregation and statistics
- Cache warming
- Database maintenance

Designed for parallel processing with progress tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of batch tasks."""
    BACKFILL_PRICES = "BACKFILL_PRICES"       # Historical price data
    BACKFILL_FUNDAMENTALS = "BACKFILL_FUND"   # Fundamental data
    CALCULATE_INDICATORS = "CALC_INDICATORS"  # Technical indicators
    TRAIN_MODEL = "TRAIN_MODEL"               # ML model training
    BACKTEST = "BACKTEST"                     # Strategy backtesting
    AGGREGATE_STATS = "AGGREGATE_STATS"       # Statistics aggregation
    WARM_CACHE = "WARM_CACHE"                 # Pre-populate caches
    CLEANUP = "CLEANUP"                       # Data cleanup
    EXPORT = "EXPORT"                         # Data export
    CUSTOM = "CUSTOM"                         # Custom task


class TaskPriority(Enum):
    """Task execution priority."""
    CRITICAL = 1   # Must complete
    HIGH = 2       # Important
    NORMAL = 3     # Standard
    LOW = 4        # Background


class TaskStatus(Enum):
    """Status of a batch task."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


@dataclass
class TaskProgress:
    """Progress tracking for a task."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0

    started_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    eta_seconds: Optional[float] = None

    current_item: str = ""
    current_step: str = ""

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        if not self.started_at or self.processed_items == 0:
            return 0.0
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed

    def update(
        self,
        processed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        current: str = ""
    ) -> None:
        """Update progress."""
        self.processed_items += processed
        self.failed_items += failed
        self.skipped_items += skipped
        self.current_item = current
        self.last_update = datetime.now()

        # Estimate ETA
        if self.items_per_second > 0:
            remaining = self.total_items - self.processed_items
            self.eta_seconds = remaining / self.items_per_second


@dataclass
class BatchTask:
    """A batch processing task."""
    id: str
    task_type: TaskType
    priority: TaskPriority
    created_at: datetime = field(default_factory=datetime.now)

    # Status
    status: TaskStatus = TaskStatus.QUEUED
    progress: TaskProgress = field(default_factory=TaskProgress)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    items: List[str] = field(default_factory=list)  # Items to process

    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: float = 3600  # 1 hour default

    # Execution
    worker_count: int = 4
    batch_size: int = 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": {
                "total": self.progress.total_items,
                "processed": self.progress.processed_items,
                "failed": self.progress.failed_items,
                "percent": self.progress.percent_complete,
            },
            "duration_seconds": self.duration_seconds,
            "error_count": len(self.errors),
        }


@dataclass
class ProcessorConfig:
    """Configuration for batch processor."""
    # Parallelism
    max_workers: int = 4
    use_multiprocessing: bool = False  # True for CPU-bound
    max_concurrent_tasks: int = 2

    # Batching
    default_batch_size: int = 100
    batch_delay_seconds: float = 0.1

    # Timeouts
    task_timeout_seconds: float = 3600  # 1 hour
    item_timeout_seconds: float = 30

    # Retry
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    # Memory
    max_memory_mb: int = 1024  # Limit memory usage
    gc_interval: int = 1000   # Run GC every N items

    # Checkpointing
    checkpoint_interval: int = 100  # Save state every N items
    checkpoint_path: str = ""


class BatchProcessor:
    """
    Handles heavy computation tasks during off-hours.

    Usage:
        processor = BatchProcessor()

        # Queue tasks
        task_id = processor.queue_task(
            TaskType.BACKFILL_PRICES,
            items=tickers,
            config={"start_date": "2024-01-01"}
        )

        # Run all tasks
        await processor.run_all()

        # Check progress
        progress = processor.get_progress(task_id)
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()

        # Task management
        self._tasks: Dict[str, BatchTask] = {}
        self._queue: List[str] = []

        # State
        self._running = False
        self._current_tasks: Set[str] = set()
        self._lock = threading.Lock()

        # Executors
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # Handlers
        self._handlers: Dict[TaskType, Callable] = {}

        # Checkpointing
        self._checkpoints: Dict[str, Dict] = {}

        # Statistics
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "items_processed": 0,
            "total_time_seconds": 0,
        }

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default task handlers."""
        self._handlers[TaskType.BACKFILL_PRICES] = self._handle_backfill_prices
        self._handlers[TaskType.CALCULATE_INDICATORS] = self._handle_calc_indicators
        self._handlers[TaskType.AGGREGATE_STATS] = self._handle_aggregate_stats
        self._handlers[TaskType.WARM_CACHE] = self._handle_warm_cache
        self._handlers[TaskType.CLEANUP] = self._handle_cleanup

    def register_handler(
        self,
        task_type: TaskType,
        handler: Callable
    ) -> None:
        """Register a custom task handler."""
        self._handlers[task_type] = handler

    def queue_task(
        self,
        task_type: TaskType,
        items: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        config: Optional[Dict] = None,
        worker_count: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        Queue a batch task.

        Args:
            task_type: Type of task
            items: Items to process
            priority: Execution priority
            config: Task-specific configuration
            worker_count: Number of parallel workers
            batch_size: Items per batch
            timeout: Task timeout in seconds

        Returns:
            Task ID
        """
        import uuid
        task_id = f"batch_{uuid.uuid4().hex[:8]}"

        task = BatchTask(
            id=task_id,
            task_type=task_type,
            priority=priority,
            items=items or [],
            config=config or {},
            worker_count=worker_count or self.config.max_workers,
            batch_size=batch_size or self.config.default_batch_size,
            timeout_seconds=timeout or self.config.task_timeout_seconds
        )

        task.progress.total_items = len(task.items)

        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
            # Sort by priority
            self._queue.sort(key=lambda x: self._tasks[x].priority.value)

        logger.info(f"Queued batch task: {task_id} ({task_type.value})")

        return task_id

    async def run_all(self) -> Dict[str, BatchTask]:
        """Run all queued tasks."""
        self._running = True
        completed = {}

        # Initialize executors
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        if self.config.use_multiprocessing:
            self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)

        try:
            while self._queue and self._running:
                # Get next task
                with self._lock:
                    if not self._queue:
                        break
                    if len(self._current_tasks) >= self.config.max_concurrent_tasks:
                        await asyncio.sleep(1)
                        continue
                    task_id = self._queue.pop(0)
                    self._current_tasks.add(task_id)

                task = self._tasks.get(task_id)
                if not task:
                    continue

                try:
                    await self._run_task(task)
                    completed[task_id] = task
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    task.status = TaskStatus.FAILED
                    task.errors.append(str(e))
                    completed[task_id] = task
                finally:
                    self._current_tasks.discard(task_id)

        finally:
            self._running = False
            if self._thread_pool:
                self._thread_pool.shutdown(wait=False)
            if self._process_pool:
                self._process_pool.shutdown(wait=False)

        return completed

    async def run_task(self, task_id: str) -> Optional[BatchTask]:
        """Run a specific task."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        # Remove from queue if present
        with self._lock:
            if task_id in self._queue:
                self._queue.remove(task_id)

        await self._run_task(task)
        return task

    async def _run_task(self, task: BatchTask) -> None:
        """Execute a batch task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.progress.started_at = datetime.now()

        logger.info(
            f"Starting batch task: {task.id} ({task.task_type.value}) "
            f"- {task.progress.total_items} items"
        )

        # Get handler
        handler = self._handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"No handler for task type: {task.task_type}")

        try:
            # Check for checkpoint
            checkpoint = self._checkpoints.get(task.id)
            start_index = 0
            if checkpoint:
                start_index = checkpoint.get("processed", 0)
                task.progress.processed_items = start_index
                logger.info(f"Resuming from checkpoint at item {start_index}")

            # Process in batches
            items = task.items[start_index:]
            for i in range(0, len(items), task.batch_size):
                if not self._running:
                    task.status = TaskStatus.PAUSED
                    break

                batch = items[i:i + task.batch_size]
                task.progress.current_step = f"Batch {i // task.batch_size + 1}"

                try:
                    # Process batch
                    batch_results = await asyncio.wait_for(
                        self._process_batch(batch, handler, task),
                        timeout=task.timeout_seconds
                    )

                    # Merge results
                    for key, value in batch_results.items():
                        if key in task.results:
                            if isinstance(task.results[key], list):
                                task.results[key].extend(value)
                            elif isinstance(task.results[key], dict):
                                task.results[key].update(value)
                        else:
                            task.results[key] = value

                except asyncio.TimeoutError:
                    logger.error(f"Batch timeout in task {task.id}")
                    task.progress.failed_items += len(batch)
                    task.errors.append(f"Batch timeout at item {i}")

                except Exception as e:
                    logger.error(f"Batch error in task {task.id}: {e}")
                    task.progress.failed_items += len(batch)
                    task.errors.append(f"Batch error: {e}")

                # Checkpoint
                if (task.progress.processed_items % self.config.checkpoint_interval) == 0:
                    self._save_checkpoint(task)

                # Rate limiting
                await asyncio.sleep(self.config.batch_delay_seconds)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.errors.append(str(e))
            raise

        # Complete
        task.completed_at = datetime.now()
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.COMPLETED

        # Update stats
        self._stats["tasks_completed"] += 1
        self._stats["items_processed"] += task.progress.processed_items
        self._stats["total_time_seconds"] += task.duration_seconds or 0

        # Clear checkpoint
        self._checkpoints.pop(task.id, None)

        logger.info(
            f"Completed batch task: {task.id} - "
            f"{task.progress.processed_items}/{task.progress.total_items} items "
            f"in {task.duration_seconds:.1f}s"
        )

    async def _process_batch(
        self,
        items: List[str],
        handler: Callable,
        task: BatchTask
    ) -> Dict[str, Any]:
        """Process a batch of items."""
        results = {}

        # Use thread pool for I/O-bound tasks
        loop = asyncio.get_event_loop()

        async def process_item(item: str) -> Tuple[str, Any]:
            """Process single item."""
            try:
                task.progress.current_item = item

                # Run handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(item, task.config)
                else:
                    result = await loop.run_in_executor(
                        self._thread_pool,
                        handler,
                        item,
                        task.config
                    )

                task.progress.update(processed=1, current=item)
                return item, result

            except Exception as e:
                task.progress.update(failed=1, current=item)
                logger.debug(f"Failed to process {item}: {e}")
                return item, None

        # Process items concurrently
        tasks = [process_item(item) for item in items]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item, result in completed:
            if result is not None and not isinstance(result, Exception):
                results[item] = result

        return results

    def _save_checkpoint(self, task: BatchTask) -> None:
        """Save task checkpoint."""
        self._checkpoints[task.id] = {
            "processed": task.progress.processed_items,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(task.results),
        }

    # Default handlers

    async def _handle_backfill_prices(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[Dict]:
        """Handle price backfill for a ticker."""
        # Would fetch historical price data
        # start_date = config.get("start_date")
        # end_date = config.get("end_date")
        return {"ticker": ticker, "status": "backfilled"}

    async def _handle_calc_indicators(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[Dict]:
        """Calculate technical indicators for a ticker."""
        # Would calculate RSI, MACD, etc.
        return {"ticker": ticker, "indicators": []}

    async def _handle_aggregate_stats(
        self,
        ticker: str,
        config: Dict
    ) -> Optional[Dict]:
        """Aggregate statistics for a ticker."""
        # Would compute various statistics
        return {"ticker": ticker, "stats": {}}

    async def _handle_warm_cache(
        self,
        key: str,
        config: Dict
    ) -> Optional[Dict]:
        """Warm cache for a key."""
        # Would pre-populate cache
        return {"key": key, "cached": True}

    async def _handle_cleanup(
        self,
        item: str,
        config: Dict
    ) -> Optional[Dict]:
        """Cleanup task handler."""
        # Would perform cleanup operations
        return {"item": item, "cleaned": True}

    # Query methods

    def get_task(self, task_id: str) -> Optional[BatchTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get task progress."""
        task = self._tasks.get(task_id)
        return task.progress if task else None

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_results(self, task_id: str) -> Optional[Dict]:
        """Get task results."""
        task = self._tasks.get(task_id)
        return task.results if task else None

    def get_pending_tasks(self) -> List[BatchTask]:
        """Get all pending tasks."""
        return [
            self._tasks[tid] for tid in self._queue
            if tid in self._tasks
        ]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            if task_id in self._queue:
                self._queue.remove(task_id)
                if task_id in self._tasks:
                    self._tasks[task_id].status = TaskStatus.CANCELLED
                return True
        return False

    def pause(self) -> None:
        """Pause processing after current batch."""
        self._running = False

    def resume(self) -> None:
        """Resume processing."""
        self._running = True

    def get_stats(self) -> Dict:
        """Get processor statistics."""
        return {
            **self._stats,
            "queued_tasks": len(self._queue),
            "running_tasks": len(self._current_tasks),
            "total_tasks": len(self._tasks),
        }

    def clear_completed(self) -> int:
        """Clear completed tasks from memory."""
        count = 0
        to_remove = []
        for task_id, task in self._tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                to_remove.append(task_id)
                count += 1
        for task_id in to_remove:
            del self._tasks[task_id]
        return count


# Singleton instance
_processor: Optional[BatchProcessor] = None


def get_batch_processor(config: Optional[ProcessorConfig] = None) -> BatchProcessor:
    """Get singleton BatchProcessor instance."""
    global _processor
    if _processor is None:
        _processor = BatchProcessor(config)
    return _processor
