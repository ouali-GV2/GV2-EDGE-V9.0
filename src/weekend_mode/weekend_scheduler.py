"""
Weekend Scheduler - Job Orchestration for Off-Hours Processing

Orchestrates all weekend processing tasks:
- Schedules scans, batch jobs, and prep tasks
- Manages execution order and dependencies
- Handles failures and retries
- Generates status reports
- Prepares system for Monday trading

Designed to run autonomously from Friday close to Monday pre-market.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
import asyncio
import logging

from .weekend_scanner import (
    WeekendScanner,
    ScanType,
    ScanPriority,
    get_weekend_scanner
)
from .monday_prep import (
    MondayPrep,
    WatchlistCategory,
    get_monday_prep
)
from .batch_processor import (
    BatchProcessor,
    TaskType,
    TaskPriority,
    get_batch_processor
)

logger = logging.getLogger(__name__)


class SchedulePhase(Enum):
    """Phases of weekend processing."""
    MARKET_CLOSE = "MARKET_CLOSE"       # Friday 4:00 PM
    FRIDAY_EVENING = "FRIDAY_EVENING"   # Friday 6:00 PM - 11:59 PM
    SATURDAY = "SATURDAY"               # All day Saturday
    SUNDAY_MORNING = "SUNDAY_MORNING"   # Sunday 6:00 AM - 12:00 PM
    SUNDAY_AFTERNOON = "SUNDAY_AFTERNOON"  # Sunday 12:00 PM - 6:00 PM
    SUNDAY_EVENING = "SUNDAY_EVENING"   # Sunday 6:00 PM - 11:59 PM
    PRE_MARKET = "PRE_MARKET"           # Monday 4:00 AM - 9:30 AM


class JobStatus(Enum):
    """Status of a scheduled job."""
    SCHEDULED = "SCHEDULED"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class JobType(Enum):
    """Types of scheduled jobs."""
    SCAN = "SCAN"           # Weekend scanner job
    BATCH = "BATCH"         # Batch processor job
    PREP = "PREP"           # Monday prep job
    REPORT = "REPORT"       # Generate report
    NOTIFY = "NOTIFY"       # Send notification
    CUSTOM = "CUSTOM"       # Custom job


@dataclass
class ScheduledJob:
    """A job scheduled for weekend execution."""
    id: str
    job_type: JobType
    phase: SchedulePhase
    priority: int = 50  # 0-100, higher = run first

    # Timing
    scheduled_time: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: float = 3600

    # Status
    status: JobStatus = JobStatus.SCHEDULED
    progress: float = 0.0  # 0-100

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Results
    result: Any = None
    error: Optional[str] = None

    # Retry
    max_retries: int = 3
    retry_count: int = 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "phase": self.phase.value,
            "status": self.status.value,
            "progress": self.progress,
            "duration": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class WeekendPlan:
    """Complete weekend execution plan."""
    created_at: datetime = field(default_factory=datetime.now)
    week_ending: date = field(default_factory=date.today)

    # Jobs by phase
    jobs_by_phase: Dict[SchedulePhase, List[ScheduledJob]] = field(
        default_factory=dict
    )

    # Overall status
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    current_phase: Optional[SchedulePhase] = None

    # Statistics
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    # Notes and reports
    notes: List[str] = field(default_factory=list)
    report: str = ""

    def get_all_jobs(self) -> List[ScheduledJob]:
        """Get all jobs in execution order."""
        all_jobs = []
        for phase in SchedulePhase:
            jobs = self.jobs_by_phase.get(phase, [])
            all_jobs.extend(sorted(jobs, key=lambda x: -x.priority))
        return all_jobs


@dataclass
class SchedulerConfig:
    """Configuration for weekend scheduler."""
    # Timing
    market_close_time: time = time(16, 0)    # 4:00 PM
    pre_market_time: time = time(4, 0)       # 4:00 AM

    # Execution
    max_concurrent_jobs: int = 3
    job_timeout_seconds: float = 3600  # 1 hour default

    # Retry
    max_retries: int = 3
    retry_delay_seconds: float = 60

    # Notifications
    notify_on_failure: bool = True
    notify_on_complete: bool = True

    # Default scans to run
    default_scans: List[ScanType] = field(default_factory=lambda: [
        ScanType.MOMENTUM_SCREEN,
        ScanType.TECHNICAL_PATTERNS,
        ScanType.EARNINGS_PREP,
        ScanType.SHORT_INTEREST,
        ScanType.SEC_FILINGS,
    ])

    # Default batch tasks
    default_batch_tasks: List[TaskType] = field(default_factory=lambda: [
        TaskType.BACKFILL_PRICES,
        TaskType.CALCULATE_INDICATORS,
        TaskType.AGGREGATE_STATS,
        TaskType.WARM_CACHE,
    ])


class WeekendScheduler:
    """
    Orchestrates all weekend processing tasks.

    Usage:
        scheduler = WeekendScheduler()

        # Create weekend plan
        plan = scheduler.create_plan()

        # Run the plan
        await scheduler.execute_plan(plan)

        # Or run in daemon mode
        await scheduler.run_daemon()
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()

        # Components
        self._scanner: WeekendScanner = get_weekend_scanner()
        self._prep: MondayPrep = get_monday_prep()
        self._processor: BatchProcessor = get_batch_processor()

        # Current plan
        self._current_plan: Optional[WeekendPlan] = None
        self._historical_plans: Dict[date, WeekendPlan] = {}

        # State
        self._running = False
        self._current_phase: Optional[SchedulePhase] = None

        # Custom job handlers
        self._handlers: Dict[str, Callable] = {}

        # Listeners
        self._listeners: List[Callable] = []

        # Statistics
        self._stats = {
            "weekends_processed": 0,
            "total_jobs_run": 0,
            "total_failures": 0,
        }

    def create_plan(
        self,
        week_ending: Optional[date] = None,
        include_scans: Optional[List[ScanType]] = None,
        include_batch: Optional[List[TaskType]] = None,
        custom_jobs: Optional[List[Dict]] = None
    ) -> WeekendPlan:
        """
        Create a weekend execution plan.

        Args:
            week_ending: Friday date for this weekend
            include_scans: Scan types to include
            include_batch: Batch task types to include
            custom_jobs: Additional custom jobs

        Returns:
            WeekendPlan with scheduled jobs
        """
        week_ending = week_ending or self._get_friday()
        include_scans = include_scans or self.config.default_scans
        include_batch = include_batch or self.config.default_batch_tasks

        plan = WeekendPlan(week_ending=week_ending)

        # Initialize phases
        for phase in SchedulePhase:
            plan.jobs_by_phase[phase] = []

        job_id = 0

        def next_id() -> str:
            nonlocal job_id
            job_id += 1
            return f"job_{job_id:03d}"

        # Phase 1: Market Close - Data cleanup and initial prep
        plan.jobs_by_phase[SchedulePhase.MARKET_CLOSE].extend([
            ScheduledJob(
                id=next_id(),
                job_type=JobType.BATCH,
                phase=SchedulePhase.MARKET_CLOSE,
                priority=90,
                config={"task_type": TaskType.CLEANUP.value}
            ),
        ])

        # Phase 2: Friday Evening - Backfill and heavy processing
        for task_type in include_batch:
            plan.jobs_by_phase[SchedulePhase.FRIDAY_EVENING].append(
                ScheduledJob(
                    id=next_id(),
                    job_type=JobType.BATCH,
                    phase=SchedulePhase.FRIDAY_EVENING,
                    priority=70,
                    config={"task_type": task_type.value}
                )
            )

        # Phase 3: Saturday - Full universe scans
        for scan_type in include_scans:
            priority = 80 if scan_type in [ScanType.MOMENTUM_SCREEN, ScanType.EARNINGS_PREP] else 60
            plan.jobs_by_phase[SchedulePhase.SATURDAY].append(
                ScheduledJob(
                    id=next_id(),
                    job_type=JobType.SCAN,
                    phase=SchedulePhase.SATURDAY,
                    priority=priority,
                    config={"scan_type": scan_type.value, "universe": "ALL"}
                )
            )

        # Phase 4: Sunday Morning - Sector analysis
        plan.jobs_by_phase[SchedulePhase.SUNDAY_MORNING].append(
            ScheduledJob(
                id=next_id(),
                job_type=JobType.SCAN,
                phase=SchedulePhase.SUNDAY_MORNING,
                priority=75,
                config={"scan_type": ScanType.SECTOR_ROTATION.value}
            )
        )

        # Phase 5: Sunday Afternoon - News sentiment
        plan.jobs_by_phase[SchedulePhase.SUNDAY_AFTERNOON].append(
            ScheduledJob(
                id=next_id(),
                job_type=JobType.SCAN,
                phase=SchedulePhase.SUNDAY_AFTERNOON,
                priority=70,
                config={"scan_type": ScanType.NEWS_SENTIMENT.value}
            )
        )

        # Phase 6: Sunday Evening - Generate Monday prep
        plan.jobs_by_phase[SchedulePhase.SUNDAY_EVENING].extend([
            ScheduledJob(
                id=next_id(),
                job_type=JobType.PREP,
                phase=SchedulePhase.SUNDAY_EVENING,
                priority=95,
                config={"generate_watchlists": True}
            ),
            ScheduledJob(
                id=next_id(),
                job_type=JobType.REPORT,
                phase=SchedulePhase.SUNDAY_EVENING,
                priority=85,
                config={"report_type": "weekend_summary"}
            ),
        ])

        # Phase 7: Pre-Market - Cache warming and notifications
        plan.jobs_by_phase[SchedulePhase.PRE_MARKET].extend([
            ScheduledJob(
                id=next_id(),
                job_type=JobType.BATCH,
                phase=SchedulePhase.PRE_MARKET,
                priority=90,
                config={"task_type": TaskType.WARM_CACHE.value}
            ),
            ScheduledJob(
                id=next_id(),
                job_type=JobType.NOTIFY,
                phase=SchedulePhase.PRE_MARKET,
                priority=100,
                config={"notification_type": "monday_ready"}
            ),
        ])

        # Add custom jobs
        if custom_jobs:
            for job_config in custom_jobs:
                phase = SchedulePhase[job_config.get("phase", "SATURDAY")]
                plan.jobs_by_phase[phase].append(
                    ScheduledJob(
                        id=next_id(),
                        job_type=JobType.CUSTOM,
                        phase=phase,
                        priority=job_config.get("priority", 50),
                        config=job_config.get("config", {})
                    )
                )

        # Count total jobs
        plan.total_jobs = sum(len(jobs) for jobs in plan.jobs_by_phase.values())

        self._current_plan = plan
        logger.info(f"Created weekend plan with {plan.total_jobs} jobs")

        return plan

    async def execute_plan(
        self,
        plan: Optional[WeekendPlan] = None,
        start_phase: Optional[SchedulePhase] = None
    ) -> WeekendPlan:
        """
        Execute a weekend plan.

        Args:
            plan: Plan to execute (uses current if None)
            start_phase: Phase to start from (for resuming)

        Returns:
            Completed plan
        """
        plan = plan or self._current_plan
        if not plan:
            plan = self.create_plan()

        plan.status = "RUNNING"
        self._running = True

        # Determine phases to execute
        phases = list(SchedulePhase)
        if start_phase:
            start_idx = phases.index(start_phase)
            phases = phases[start_idx:]

        try:
            for phase in phases:
                if not self._running:
                    break

                plan.current_phase = phase
                self._current_phase = phase

                jobs = plan.jobs_by_phase.get(phase, [])
                if not jobs:
                    continue

                logger.info(f"Starting phase: {phase.value} ({len(jobs)} jobs)")

                # Sort by priority (highest first)
                jobs.sort(key=lambda x: -x.priority)

                # Execute jobs
                for job in jobs:
                    if not self._running:
                        break

                    # Check dependencies
                    if not self._check_dependencies(job, plan):
                        job.status = JobStatus.SKIPPED
                        logger.warning(f"Skipping job {job.id}: dependencies not met")
                        continue

                    await self._execute_job(job, plan)

                logger.info(f"Completed phase: {phase.value}")

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = "FAILED"
            plan.notes.append(f"Execution failed: {e}")
            raise

        finally:
            self._running = False
            plan.current_phase = None

        # Finalize plan
        if plan.failed_jobs == 0:
            plan.status = "COMPLETED"
        else:
            plan.status = f"COMPLETED_WITH_ERRORS ({plan.failed_jobs} failures)"

        # Generate report
        plan.report = self._generate_report(plan)

        # Store in history
        self._historical_plans[plan.week_ending] = plan

        # Update stats
        self._stats["weekends_processed"] += 1
        self._stats["total_jobs_run"] += plan.completed_jobs
        self._stats["total_failures"] += plan.failed_jobs

        return plan

    async def _execute_job(
        self,
        job: ScheduledJob,
        plan: WeekendPlan
    ) -> None:
        """Execute a single job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        logger.info(f"Executing job: {job.id} ({job.job_type.value})")

        try:
            # Route to appropriate handler
            if job.job_type == JobType.SCAN:
                await self._execute_scan_job(job)
            elif job.job_type == JobType.BATCH:
                await self._execute_batch_job(job)
            elif job.job_type == JobType.PREP:
                await self._execute_prep_job(job)
            elif job.job_type == JobType.REPORT:
                await self._execute_report_job(job, plan)
            elif job.job_type == JobType.NOTIFY:
                await self._execute_notify_job(job, plan)
            elif job.job_type == JobType.CUSTOM:
                await self._execute_custom_job(job)

            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            plan.completed_jobs += 1

            logger.info(f"Completed job: {job.id} in {job.duration_seconds:.1f}s")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            job.error = str(e)

            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.SCHEDULED
                logger.info(f"Retrying job {job.id} (attempt {job.retry_count})")
                await asyncio.sleep(self.config.retry_delay_seconds)
                await self._execute_job(job, plan)
            else:
                job.status = JobStatus.FAILED
                plan.failed_jobs += 1

        finally:
            job.completed_at = datetime.now()

    async def _execute_scan_job(self, job: ScheduledJob) -> None:
        """Execute a scan job."""
        scan_type = ScanType[job.config.get("scan_type", "MOMENTUM_SCREEN")]
        universe = job.config.get("universe", "ALL")

        job_id = self._scanner.queue_scan(
            scan_type=scan_type,
            universe=universe,
            priority=ScanPriority.HIGH
        )

        # Run scan
        await self._scanner.run_job(job_id)

        # Get results
        scanner_job = self._scanner.get_job(job_id)
        if scanner_job:
            job.result = {
                "candidates": scanner_job.candidates,
                "processed": scanner_job.processed_tickers,
            }

    async def _execute_batch_job(self, job: ScheduledJob) -> None:
        """Execute a batch processing job."""
        task_type = TaskType[job.config.get("task_type", "AGGREGATE_STATS")]
        items = job.config.get("items", [])

        task_id = self._processor.queue_task(
            task_type=task_type,
            items=items,
            priority=TaskPriority.HIGH
        )

        # Run task
        await self._processor.run_task(task_id)

        # Get results
        task = self._processor.get_task(task_id)
        if task:
            job.result = task.results

    async def _execute_prep_job(self, job: ScheduledJob) -> None:
        """Execute Monday prep job."""
        # Collect all candidates from scanner
        candidates = self._scanner.get_candidates()

        # Add to prep
        for ticker in candidates:
            results = self._scanner.get_results(ticker)
            if results:
                best_result = max(results, key=lambda x: x.overall_score)

                # Determine category
                category = WatchlistCategory.MOMENTUM
                if best_result.scan_type == ScanType.EARNINGS_PREP:
                    category = WatchlistCategory.EARNINGS
                elif best_result.scan_type == ScanType.SHORT_INTEREST:
                    category = WatchlistCategory.SQUEEZE
                elif best_result.scan_type == ScanType.TECHNICAL_PATTERNS:
                    category = WatchlistCategory.TECHNICAL

                self._prep.add_candidate(
                    ticker=ticker,
                    category=category,
                    overall_score=best_result.overall_score,
                    momentum_score=best_result.momentum_score,
                    signals=best_result.signals,
                    catalysts=best_result.catalysts,
                    risk_factors=best_result.risk_flags
                )

        # Generate plan
        monday_plan = self._prep.generate_plan()
        job.result = monday_plan.to_dict()

    async def _execute_report_job(
        self,
        job: ScheduledJob,
        plan: WeekendPlan
    ) -> None:
        """Generate weekend report."""
        report_type = job.config.get("report_type", "weekend_summary")

        if report_type == "weekend_summary":
            report = self._generate_report(plan)
        else:
            report = f"Report type: {report_type}"

        job.result = {"report": report}

    async def _execute_notify_job(
        self,
        job: ScheduledJob,
        plan: WeekendPlan
    ) -> None:
        """Send notifications."""
        notification_type = job.config.get("notification_type")

        if notification_type == "monday_ready":
            message = f"Weekend processing complete. Ready for Monday trading."
            # Would integrate with notification system
            logger.info(f"Notification: {message}")

        job.result = {"notified": True}

    async def _execute_custom_job(self, job: ScheduledJob) -> None:
        """Execute custom job."""
        handler_name = job.config.get("handler")
        if handler_name and handler_name in self._handlers:
            result = await self._handlers[handler_name](job.config)
            job.result = result
        else:
            raise ValueError(f"No handler for custom job: {handler_name}")

    def _check_dependencies(
        self,
        job: ScheduledJob,
        plan: WeekendPlan
    ) -> bool:
        """Check if job dependencies are met."""
        if not job.depends_on:
            return True

        all_jobs = plan.get_all_jobs()
        job_map = {j.id: j for j in all_jobs}

        for dep_id in job.depends_on:
            dep_job = job_map.get(dep_id)
            if not dep_job or dep_job.status != JobStatus.COMPLETED:
                return False

        return True

    def _generate_report(self, plan: WeekendPlan) -> str:
        """Generate weekend summary report."""
        lines = [
            "=" * 50,
            "WEEKEND PROCESSING REPORT",
            "=" * 50,
            f"Week Ending: {plan.week_ending}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Status: {plan.status}",
            "",
            f"Jobs: {plan.completed_jobs}/{plan.total_jobs} completed",
            f"Failures: {plan.failed_jobs}",
            "",
            "PHASE SUMMARY:",
            "-" * 30,
        ]

        for phase in SchedulePhase:
            jobs = plan.jobs_by_phase.get(phase, [])
            if not jobs:
                continue

            completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
            lines.append(f"  {phase.value}: {completed}/{len(jobs)} ({failed} failed)")

        # Monday prep summary
        monday_plan = self._prep.get_current_plan()
        if monday_plan:
            lines.extend([
                "",
                "MONDAY PREP:",
                "-" * 30,
                f"  Primary Focus: {len(monday_plan.primary_focus)} tickers",
                f"  Secondary Focus: {len(monday_plan.secondary_focus)} tickers",
                f"  Avoid List: {len(monday_plan.avoid_list)} tickers",
                "",
                f"  Top Picks: {', '.join(monday_plan.primary_focus[:5])}",
            ])

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)

    def _get_friday(self) -> date:
        """Get the Friday date for this week."""
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0 and datetime.now().time() > self.config.market_close_time:
            days_until_friday = 7
        return today + timedelta(days=days_until_friday)

    # Control methods

    def stop(self) -> None:
        """Stop execution after current job."""
        self._running = False

    def register_handler(
        self,
        name: str,
        handler: Callable
    ) -> None:
        """Register a custom job handler."""
        self._handlers[name] = handler

    def add_listener(self, callback: Callable) -> None:
        """Add job completion listener."""
        self._listeners.append(callback)

    # Query methods

    def get_current_plan(self) -> Optional[WeekendPlan]:
        """Get current plan."""
        return self._current_plan

    def get_plan_history(self) -> Dict[date, WeekendPlan]:
        """Get historical plans."""
        return self._historical_plans.copy()

    def get_current_phase(self) -> Optional[SchedulePhase]:
        """Get current execution phase."""
        return self._current_phase

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            **self._stats,
            "is_running": self._running,
            "current_phase": self._current_phase.value if self._current_phase else None,
        }


# Singleton instance
_scheduler: Optional[WeekendScheduler] = None


def get_weekend_scheduler(config: Optional[SchedulerConfig] = None) -> WeekendScheduler:
    """Get singleton WeekendScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = WeekendScheduler(config)
    return _scheduler
