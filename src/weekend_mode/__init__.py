"""
Weekend Mode - Strategic Preparation for Monday Trading

Components:
- WeekendScanner: Comprehensive batch scanning during off-hours
- MondayPrep: Strategic watchlist generation
- BatchProcessor: Heavy computation tasks
- WeekendScheduler: Orchestrates all weekend processing

Usage:
    from src.weekend_mode import get_weekend_scheduler, SchedulePhase

    # Create and run weekend plan
    scheduler = get_weekend_scheduler()
    plan = scheduler.create_plan()
    await scheduler.execute_plan(plan)

    # Get Monday prep results
    from src.weekend_mode import get_monday_prep
    prep = get_monday_prep()
    monday_plan = prep.get_current_plan()
    print(f"Primary focus: {monday_plan.primary_focus}")

Execution phases:
- MARKET_CLOSE: Friday 4:00 PM - Initial cleanup
- FRIDAY_EVENING: Backfill and heavy processing
- SATURDAY: Full universe scans
- SUNDAY_MORNING: Sector analysis
- SUNDAY_AFTERNOON: News sentiment
- SUNDAY_EVENING: Generate Monday prep
- PRE_MARKET: Monday 4:00 AM - Cache warming, notifications
"""

# Weekend Scanner
from .weekend_scanner import (
    WeekendScanner,
    ScannerConfig,
    ScanJob,
    ScanResult,
    ScanType,
    ScanPriority,
    ScanStatus,
    get_weekend_scanner,
)

# Monday Prep
from .monday_prep import (
    MondayPrep,
    MondayPrepConfig,
    MondayPlan,
    WatchlistItem,
    WatchlistCategory,
    WatchlistPriority,
    get_monday_prep,
)

# Batch Processor
from .batch_processor import (
    BatchProcessor,
    ProcessorConfig,
    BatchTask,
    TaskProgress,
    TaskType,
    TaskPriority,
    TaskStatus,
    get_batch_processor,
)

# Weekend Scheduler
from .weekend_scheduler import (
    WeekendScheduler,
    SchedulerConfig,
    WeekendPlan,
    ScheduledJob,
    SchedulePhase,
    JobType,
    JobStatus,
    get_weekend_scheduler,
)

__all__ = [
    # Scanner
    "WeekendScanner",
    "ScannerConfig",
    "ScanJob",
    "ScanResult",
    "ScanType",
    "ScanPriority",
    "ScanStatus",
    "get_weekend_scanner",
    # Monday Prep
    "MondayPrep",
    "MondayPrepConfig",
    "MondayPlan",
    "WatchlistItem",
    "WatchlistCategory",
    "WatchlistPriority",
    "get_monday_prep",
    # Batch Processor
    "BatchProcessor",
    "ProcessorConfig",
    "BatchTask",
    "TaskProgress",
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    "get_batch_processor",
    # Scheduler
    "WeekendScheduler",
    "SchedulerConfig",
    "WeekendPlan",
    "ScheduledJob",
    "SchedulePhase",
    "JobType",
    "JobStatus",
    "get_weekend_scheduler",
]
