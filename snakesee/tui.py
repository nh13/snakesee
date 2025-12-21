"""Rich TUI for Snakemake workflow monitoring."""

import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from snakesee.estimator import TimeEstimator
from snakesee.events import EventReader
from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.events import get_event_file_path
from snakesee.models import JobInfo
from snakesee.models import RuleTimingStats
from snakesee.models import ThreadTimingStats
from snakesee.models import TimeEstimate
from snakesee.models import WeightingStrategy
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.models import format_duration
from snakesee.parser import IncrementalLogReader
from snakesee.parser import parse_workflow_state
from snakesee.plugins import find_rule_log
from snakesee.plugins import parse_tool_progress
from snakesee.plugins.base import ToolProgress
from snakesee.validation import EventAccumulator
from snakesee.validation import ValidationLogger
from snakesee.validation import compare_states

# Refresh rate bounds
MIN_REFRESH_RATE = 0.5
MAX_REFRESH_RATE = 60.0
DEFAULT_REFRESH_RATE = 1.0

# Fulcrum Genomics brand colors
FG_BLUE = "#26a8e0"
FG_GREEN = "#38b44a"

# Fulcrum Genomics logo path (easter egg)
FG_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


class LayoutMode(Enum):
    """Available TUI layout modes."""

    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"


class WorkflowMonitorTUI:
    """
    Rich TUI for monitoring Snakemake workflows.

    Provides a full-screen terminal interface with:
    - Progress bar showing overall completion
    - Table of currently running jobs
    - Recent completions
    - Time estimation with confidence bounds

    Keyboard Controls (vim-like):
        q: Quit
        ?: Show help
        p: Pause/resume auto-refresh
        e: Toggle time estimation
        w: Toggle wildcard conditioning (estimate per sample/batch)
        r: Force refresh
        Ctrl+r: Hard refresh (reload historical data)

        Refresh rate (vim-style):
        h: Decrease by 5s (faster)
        j: Decrease by 0.5s (faster)
        k: Increase by 0.5s (slower)
        l: Increase by 5s (slower)
        0: Reset to default (1s)
        G: Set to minimum (0.5s, fastest)

        Layout:
        Tab: Cycle layout mode (full/compact/minimal)

        Filter:
        /: Enter filter mode (filter rules by name)
        n: Next filter match
        N: Previous filter match
        Esc: Clear filter, return to latest log

        Log Navigation:
        [: View older log/execution (1 step)
        ]: View newer log/execution (1 step)
        {: View older log/execution (5 steps)
        }: View newer log/execution (5 steps)

        Table Sorting:
        s: Cycle sort table forward (Running -> Completions -> Pending -> Stats -> none)
        S: Cycle sort table backward
        1-4: Sort by column (press again to reverse)

    Attributes:
        workflow_dir: Path to the workflow directory.
        refresh_rate: How often to refresh the display (seconds).
        use_estimation: Whether to use historical time estimation.
    """

    def __init__(
        self,
        workflow_dir: Path,
        refresh_rate: float = DEFAULT_REFRESH_RATE,
        use_estimation: bool = True,
        profile_path: Path | None = None,
        use_wildcard_conditioning: bool = False,
        weighting_strategy: WeightingStrategy = "index",
        half_life_logs: int = 10,
        half_life_days: float = 7.0,
    ) -> None:
        """
        Initialize the TUI.

        Args:
            workflow_dir: Path to workflow directory containing .snakemake/.
            refresh_rate: Refresh interval in seconds.
            use_estimation: Whether to enable time estimation.
            profile_path: Optional path to a timing profile for bootstrapping estimates.
            use_wildcard_conditioning: Whether to enable wildcard-conditioned estimates.
            weighting_strategy: Strategy for weighting historical data ("index" or "time").
            half_life_logs: Half-life in run count for index-based weighting.
            half_life_days: Half-life in days for time-based weighting.
        """
        self.workflow_dir = workflow_dir
        self.refresh_rate = refresh_rate
        self.use_estimation = use_estimation
        self.profile_path = profile_path
        self.weighting_strategy = weighting_strategy
        self.half_life_logs = half_life_logs
        self.half_life_days = half_life_days
        self.console = Console()
        self._running = True
        self._estimator: TimeEstimator | None = None
        self._force_refresh = False

        # Wildcard conditioning toggle
        self._use_wildcard_conditioning: bool = use_wildcard_conditioning

        # New state for vim-like features
        self._paused: bool = False
        self._show_help: bool = False
        self._layout_mode: LayoutMode = LayoutMode.FULL
        self._filter_text: str | None = None
        self._filter_mode: bool = False
        self._filter_input = ""
        self._filter_matches: list[str] = []
        self._filter_index = 0

        # Log file navigation
        self._available_logs: list[Path] = []
        self._current_log_index: int = 0  # 0 = most recent
        self._refresh_log_list()

        # Table sorting state
        self._sort_table: str | None = None  # "running", "completions", "pending", or "stats"
        self._sort_column: int = 0  # 0-indexed column
        self._sort_ascending: bool = True

        # Easter egg state
        self._easter_egg_pending_f = False  # True if 'f' was just pressed
        self._show_easter_egg = False

        # Cutoff time for historical view (updated in _poll_state)
        self._cutoff_time: float | None = None

        # Job log viewer state
        self._job_selection_mode: bool = False  # True when a job is selected
        self._log_source: str | None = None  # "running" or "completions"
        self._selected_job_index: int = 0  # Index into running_jobs list
        self._selected_completion_index: int = 0  # Index into completions list
        self._log_scroll_offset: int = 0  # Lines to skip from end (0 = show latest)
        self._cached_log_path: Path | None = None
        self._cached_log_lines: list[str] = []
        self._cached_log_mtime: float = 0

        # Event reader for real-time events from logger plugin
        self._event_reader: EventReader | None = None
        self._events_enabled: bool = True
        self._init_event_reader()

        # Incremental log reader for efficient polling
        self._log_reader: IncrementalLogReader | None = None
        self._init_log_reader()

        # Validation: compare event-based state with parsed state
        self._event_accumulator: EventAccumulator | None = None
        self._validation_logger: ValidationLogger | None = None
        self._init_validation()

        # Track job_ids already added to rule_stats (to avoid duplicates)
        self._rule_stats_job_ids: set[str] = set()

        # Track threads by job_id (populated from JOB_SUBMITTED events)
        self._job_threads: dict[str, int] = {}

        # Thread-grouped timing stats: rule -> ThreadTimingStats
        self._thread_stats: dict[str, ThreadTimingStats] = {}

        self._init_estimator()

    def _refresh_log_list(self) -> None:
        """Refresh the list of available log files."""
        log_dir = self.workflow_dir / ".snakemake" / "log"
        if log_dir.exists():
            # Sort by modification time, newest first
            logs = sorted(
                log_dir.glob("*.snakemake.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            self._available_logs = logs
        else:
            self._available_logs = []

        # Reset to most recent if current index is out of bounds
        if self._current_log_index >= len(self._available_logs):
            self._current_log_index = 0

    def _get_current_log(self) -> Path | None:
        """Get the currently selected log file."""
        if not self._available_logs:
            return None
        if self._current_log_index < len(self._available_logs):
            return self._available_logs[self._current_log_index]
        return self._available_logs[0] if self._available_logs else None

    def _init_estimator(self) -> None:
        """Initialize or reinitialize the time estimator."""
        self._rule_stats_job_ids.clear()
        self._job_threads.clear()
        self._thread_stats.clear()
        if self.use_estimation:
            self._estimator = TimeEstimator(
                use_wildcard_conditioning=self._use_wildcard_conditioning,
                weighting_strategy=self.weighting_strategy,
                half_life_logs=self.half_life_logs,
                half_life_days=self.half_life_days,
            )

            # Load from profile first if available
            if self.profile_path is not None and self.profile_path.exists():
                try:
                    from snakesee.profile import load_profile

                    profile = load_profile(self.profile_path)
                    self._estimator.rule_stats = profile.to_rule_stats()
                except (OSError, ValueError):
                    pass  # Fall back to metadata only

            # Load/merge with live metadata (live data takes precedence for recent runs)
            metadata_dir = self.workflow_dir / ".snakemake" / "metadata"
            if metadata_dir.exists():
                self._estimator.load_from_metadata(metadata_dir)

            # Initialize thread stats from log parsing (metadata doesn't have threads)
            self._init_thread_stats_from_log()

            # Share thread stats with estimator for thread-aware ETA
            self._estimator.thread_stats = self._thread_stats
        else:
            self._estimator = None

    def _init_thread_stats_from_log(self) -> None:
        """Initialize thread stats from all log files (metadata doesn't have threads)."""
        from snakesee.parser import find_all_logs
        from snakesee.parser import parse_completed_jobs_from_log

        snakemake_dir = self.workflow_dir / ".snakemake"
        log_paths = find_all_logs(snakemake_dir)
        if not log_paths:
            return

        for log_path in log_paths:
            for job in parse_completed_jobs_from_log(log_path):
                if job.threads is None or job.duration is None:
                    continue
                if job.rule not in self._thread_stats:
                    self._thread_stats[job.rule] = ThreadTimingStats(rule=job.rule)
                thread_stats = self._thread_stats[job.rule]
                if job.threads not in thread_stats.stats_by_threads:
                    thread_stats.stats_by_threads[job.threads] = RuleTimingStats(rule=job.rule)
                ts = thread_stats.stats_by_threads[job.threads]
                ts.durations.append(job.duration)
                if job.end_time is not None:
                    ts.timestamps.append(job.end_time)
                if job.input_size is not None:
                    ts.input_sizes.append(job.input_size)

    def _init_event_reader(self) -> None:
        """Initialize the event reader if event file exists."""
        if not self._events_enabled:
            return

        event_file = get_event_file_path(self.workflow_dir)
        if event_file.exists():
            self._event_reader = EventReader(event_file)
        else:
            self._event_reader = None

    def _init_log_reader(self) -> None:
        """Initialize the incremental log reader.

        Creates a reader for the current log file, enabling efficient
        incremental parsing instead of re-reading the entire file on each poll.
        """
        from snakesee.parser import find_latest_log

        snakemake_dir = self.workflow_dir / ".snakemake"
        log_path = find_latest_log(snakemake_dir)
        if log_path is not None:
            self._log_reader = IncrementalLogReader(log_path)
        else:
            # Create with a placeholder path; will be updated when log appears
            self._log_reader = IncrementalLogReader(
                snakemake_dir / "log" / "placeholder.snakemake.log"
            )

    def _init_validation(self) -> None:
        """Initialize validation if event file exists.

        Validation is automatically enabled when the logger plugin's event
        file is detected, allowing comparison between event-based and
        parsed state to find bugs in either approach.
        """
        event_file = get_event_file_path(self.workflow_dir)
        if event_file.exists():
            self._event_accumulator = EventAccumulator()
            self._validation_logger = ValidationLogger(self.workflow_dir)
            self._validation_logger.log_session_start()

    def _validate_state(self, events: list[SnakeseeEvent], parsed: WorkflowProgress) -> None:
        """Compare event-based state with parsed state and log discrepancies.

        Args:
            events: New events to process.
            parsed: Current parsed workflow progress.
        """
        # Initialize validation if not yet done (event file may have appeared)
        if self._event_accumulator is None:
            self._init_validation()

        if self._event_accumulator is None or self._validation_logger is None:
            return

        # Accumulate new events
        self._event_accumulator.process_events(events)

        # Only compare if we have meaningful state from events
        if not self._event_accumulator.workflow_started:
            return

        # Compare states and log discrepancies
        discrepancies = compare_states(self._event_accumulator, parsed)

        if discrepancies:
            self._validation_logger.log_discrepancies(discrepancies)

        # Log summary periodically (every comparison for now)
        self._validation_logger.log_summary(self._event_accumulator, parsed)

    def _read_new_events(self) -> list[SnakeseeEvent]:
        """Read new events from the event file if available.

        Returns:
            List of new events, or empty list if no events or event reading disabled.
        """
        if not self._events_enabled or self._event_reader is None:
            # Try to initialize if event file now exists
            if self._events_enabled and self._event_reader is None:
                event_file = get_event_file_path(self.workflow_dir)
                if event_file.exists():
                    self._event_reader = EventReader(event_file)

            if self._event_reader is None:
                return []

        return self._event_reader.read_new_events()

    def _handle_job_submitted_event(
        self,
        event: SnakeseeEvent,
        running_jobs: list[JobInfo],
    ) -> None:
        """Handle JOB_SUBMITTED event - capture threads info."""
        if event.job_id is None:
            return
        job_id_str = str(event.job_id)
        if event.threads is not None:
            self._job_threads[job_id_str] = event.threads
        threads = event.threads or self._job_threads.get(job_id_str)
        for i, job in enumerate(running_jobs):
            if job.job_id == job_id_str:
                running_jobs[i] = JobInfo(
                    rule=job.rule,
                    job_id=job.job_id,
                    start_time=job.start_time,
                    end_time=job.end_time,
                    output_file=job.output_file,
                    wildcards=event.wildcards_dict or job.wildcards,
                    input_size=job.input_size,
                    threads=threads,
                )
                break

    def _handle_job_started_event(
        self,
        event: SnakeseeEvent,
        running_jobs: list[JobInfo],
    ) -> None:
        """Handle JOB_STARTED event - update start time and threads."""
        if event.job_id is None:
            return
        job_id_str = str(event.job_id)
        threads = event.threads or self._job_threads.get(job_id_str)
        for i, job in enumerate(running_jobs):
            if job.job_id == job_id_str:
                running_jobs[i] = JobInfo(
                    rule=job.rule,
                    job_id=job.job_id,
                    start_time=event.timestamp,
                    end_time=job.end_time,
                    output_file=job.output_file,
                    wildcards=job.wildcards,
                    input_size=job.input_size,
                    threads=threads or job.threads,
                )
                break

    def _handle_job_finished_event(
        self,
        event: SnakeseeEvent,
        completions: list[JobInfo],
    ) -> None:
        """Handle JOB_FINISHED event - update completion with accurate duration."""
        if event.job_id is None or event.duration is None:
            return
        job_id_str = str(event.job_id)
        threads = event.threads or self._job_threads.get(job_id_str)
        for i, job in enumerate(completions):
            if job.job_id == job_id_str:
                completions[i] = JobInfo(
                    rule=job.rule,
                    job_id=job.job_id,
                    start_time=event.timestamp - event.duration
                    if event.duration
                    else job.start_time,
                    end_time=event.timestamp,
                    output_file=job.output_file,
                    wildcards=job.wildcards,
                    input_size=job.input_size,
                    threads=threads or job.threads,
                )
                break

    def _handle_job_error_event(
        self,
        event: SnakeseeEvent,
        failed_list: list[JobInfo],
    ) -> int:
        """Handle JOB_ERROR event - track failed job. Returns new failed count."""
        if event.job_id is None:
            return len(failed_list)
        job_id_str = str(event.job_id)
        if not any(j.job_id == job_id_str for j in failed_list):
            failed_list.append(
                JobInfo(
                    rule=event.rule_name or "unknown",
                    job_id=job_id_str,
                    start_time=event.timestamp - event.duration if event.duration else None,
                    end_time=event.timestamp,
                    wildcards=event.wildcards_dict,
                    threads=event.threads,
                )
            )
        return len(failed_list)

    def _apply_events_to_progress(
        self, progress: WorkflowProgress, events: list[SnakeseeEvent]
    ) -> WorkflowProgress:
        """Apply event updates to enhance progress accuracy.

        Events from the logger plugin provide more accurate timing and
        status information than log parsing.

        Args:
            progress: The current workflow progress from parsing.
            events: New events from the logger plugin.

        Returns:
            Updated WorkflowProgress with event data applied.
        """
        if not events:
            return progress

        # Track updates from events
        new_total = progress.total_jobs
        new_completed = progress.completed_jobs
        new_running_jobs = list(progress.running_jobs)
        new_completions = list(progress.recent_completions)
        new_failed = progress.failed_jobs
        new_failed_list = list(progress.failed_jobs_list)

        # Process events to update state
        for event in events:
            if event.event_type == EventType.PROGRESS:
                if event.total_jobs is not None:
                    new_total = event.total_jobs
                if event.completed_jobs is not None:
                    new_completed = event.completed_jobs
            elif event.event_type == EventType.JOB_SUBMITTED:
                self._handle_job_submitted_event(event, new_running_jobs)
            elif event.event_type == EventType.JOB_STARTED:
                self._handle_job_started_event(event, new_running_jobs)
            elif event.event_type == EventType.JOB_FINISHED:
                self._handle_job_finished_event(event, new_completions)
            elif event.event_type == EventType.JOB_ERROR:
                new_failed = self._handle_job_error_event(event, new_failed_list)

        # Return updated progress
        return WorkflowProgress(
            workflow_dir=progress.workflow_dir,
            status=progress.status,
            total_jobs=new_total,
            completed_jobs=new_completed,
            failed_jobs=new_failed,
            failed_jobs_list=new_failed_list,
            running_jobs=new_running_jobs,
            recent_completions=new_completions,
            start_time=progress.start_time,
            log_file=progress.log_file,
        )

    def _update_rule_stats_from_completions(self, progress: WorkflowProgress) -> None:
        """Update rule_stats with newly completed jobs from recent_completions.

        This ensures the Rule Statistics panel shows data from jobs that completed
        during this monitoring session, not just historical data from startup.
        """
        if self._estimator is None:
            return

        for job in progress.recent_completions:
            # Create a unique key for deduplication
            # Use job_id if available, otherwise use (rule, end_time) tuple
            if job.job_id is not None:
                dedup_key = f"id:{job.job_id}"
            elif job.end_time is not None:
                dedup_key = f"time:{job.rule}:{int(job.end_time)}"
            else:
                # Can't deduplicate without job_id or end_time, skip
                continue

            if dedup_key in self._rule_stats_job_ids:
                continue

            # Skip if we don't have a valid duration
            duration = job.duration
            if duration is None:
                continue

            # Add to rule_stats
            if job.rule not in self._estimator.rule_stats:
                self._estimator.rule_stats[job.rule] = RuleTimingStats(rule=job.rule)

            stats = self._estimator.rule_stats[job.rule]
            stats.durations.append(duration)
            if job.end_time is not None:
                stats.timestamps.append(job.end_time)
            if job.input_size is not None:
                stats.input_sizes.append(job.input_size)

            # Also update thread stats if thread info is available
            if job.threads is not None:
                if job.rule not in self._thread_stats:
                    self._thread_stats[job.rule] = ThreadTimingStats(rule=job.rule)
                thread_stats = self._thread_stats[job.rule]
                if job.threads not in thread_stats.stats_by_threads:
                    thread_stats.stats_by_threads[job.threads] = RuleTimingStats(rule=job.rule)
                ts = thread_stats.stats_by_threads[job.threads]
                ts.durations.append(duration)
                if job.end_time is not None:
                    ts.timestamps.append(job.end_time)
                if job.input_size is not None:
                    ts.input_sizes.append(job.input_size)

            # Mark this job as processed
            self._rule_stats_job_ids.add(dedup_key)

    def _handle_easter_egg_key(self, key: str) -> bool | None:
        """Handle easter egg keys. Returns True/False if handled, None to continue."""
        # Handle easter egg display (any key dismisses)
        if self._show_easter_egg:
            self._show_easter_egg = False
            self._force_refresh = True
            return False

        # Easter egg: 'f' followed by 'g'
        if self._easter_egg_pending_f:
            self._easter_egg_pending_f = False
            if key.lower() == "g":
                self._show_easter_egg = True
                self._force_refresh = True
                return False
        if key.lower() == "f":
            self._easter_egg_pending_f = True
            # Don't return - let other handlers process 'f' if needed

        return None  # Not handled, continue processing

    def _handle_key(self, key: str) -> bool:
        """
        Handle a keypress.

        Args:
            key: The key that was pressed.

        Returns:
            True if should quit, False otherwise.
        """
        # Handle job selection mode (before filter mode)
        if self._job_selection_mode:
            # Use large number; actual bounds checked in _make_job_log_panel
            return self._handle_job_selection_key(key, 1000)

        # Handle filter input mode
        if self._filter_mode:
            return self._handle_filter_key(key)

        # Handle easter egg
        easter_result = self._handle_easter_egg_key(key)
        if easter_result is not None:
            return easter_result

        # Handle help overlay
        if self._show_help:
            # Any key closes help
            self._show_help = False
            self._force_refresh = True
            return False

        # Normal mode keybindings
        if key.lower() == "q":
            return True

        # Dispatch to specialized handlers
        if self._handle_toggle_key(key):
            return False
        if self._handle_refresh_rate_key(key):
            return False
        if self._handle_navigation_key(key):
            return False
        if self._handle_sort_key(key):
            return False
        if self._handle_log_navigation_key(key):
            return False

        return False

    def _handle_toggle_key(self, key: str) -> bool:
        """Handle toggle keys (?, p, e, w, r, Ctrl+r). Returns True if key was handled."""
        if key == "?":
            self._show_help = True
            self._force_refresh = True
            return True
        if key.lower() == "p":
            self._paused = not self._paused
            self._force_refresh = True
            return True
        if key.lower() == "e":
            self.use_estimation = not self.use_estimation
            self._init_estimator()
            self._force_refresh = True
            return True
        if key.lower() == "w":
            # Toggle wildcard conditioning
            self._use_wildcard_conditioning = not self._use_wildcard_conditioning
            self._init_estimator()
            self._force_refresh = True
            return True
        if key.lower() == "r":
            self._force_refresh = True
            return True
        if key == "\x12":  # Ctrl+r - hard refresh
            self._init_estimator()
            self._force_refresh = True
            return True
        return False

    def _handle_refresh_rate_key(self, key: str) -> bool:
        """Handle refresh rate keys (h/j/k/l, 0, G). Returns True if key was handled."""
        if key == "j":
            self.refresh_rate = max(MIN_REFRESH_RATE, self.refresh_rate - 0.5)
            self._force_refresh = True
            return True
        if key == "k":
            self.refresh_rate = min(MAX_REFRESH_RATE, self.refresh_rate + 0.5)
            self._force_refresh = True
            return True
        if key == "h":
            self.refresh_rate = max(MIN_REFRESH_RATE, self.refresh_rate - 5.0)
            self._force_refresh = True
            return True
        if key == "l":
            self.refresh_rate = min(MAX_REFRESH_RATE, self.refresh_rate + 5.0)
            self._force_refresh = True
            return True
        if key == "0":
            self.refresh_rate = DEFAULT_REFRESH_RATE
            self._force_refresh = True
            return True
        if key == "G":
            self.refresh_rate = MIN_REFRESH_RATE
            self._force_refresh = True
            return True
        return False

    def _handle_navigation_key(self, key: str) -> bool:
        """Handle navigation keys (Tab, /, n, N, Esc, Enter). Returns True if handled."""
        if key == "\r" or key == "\n":  # Enter - toggle job selection mode
            self._job_selection_mode = not self._job_selection_mode
            if self._job_selection_mode:
                # Entering selection mode - default to running jobs
                self._log_source = "running"
            else:
                # Exiting selection mode - reset state
                self._log_source = None
                self._selected_job_index = 0
                self._selected_completion_index = 0
                self._log_scroll_offset = 0
                self._cached_log_path = None
                self._cached_log_lines = []
            self._force_refresh = True
            return True
        if key == "\t":  # Tab - cycle layout
            modes = list(LayoutMode)
            current_idx = modes.index(self._layout_mode)
            self._layout_mode = modes[(current_idx + 1) % len(modes)]
            self._force_refresh = True
            return True
        if key == "/":  # Enter filter mode
            self._filter_mode = True
            self._filter_input = ""
            self._force_refresh = True
            return True
        if key == "n" and self._filter_matches:
            self._filter_index = (self._filter_index + 1) % len(self._filter_matches)
            self._force_refresh = True
            return True
        if key == "N" and self._filter_matches:
            self._filter_index = (self._filter_index - 1) % len(self._filter_matches)
            self._force_refresh = True
            return True
        if key == "\x1b":  # Escape - clear filter and return to latest log
            self._filter_text = None
            self._filter_matches = []
            self._filter_index = 0
            self._current_log_index = 0  # Return to latest
            self._force_refresh = True
            return True
        return False

    def _handle_sort_key(self, key: str) -> bool:
        """Handle table sorting keys (s/S, 1-4). Returns True if handled."""
        if key == "s":
            # Cycle forward: None -> running -> completions -> pending -> stats -> None
            cycle = [None, "running", "completions", "pending", "stats"]
            current_idx = cycle.index(self._sort_table) if self._sort_table in cycle else 0
            self._sort_table = cycle[(current_idx + 1) % len(cycle)]
            self._sort_column = 0
            self._sort_ascending = True
            self._force_refresh = True
            return True
        if key == "S":
            # Cycle backward: None -> stats -> pending -> completions -> running -> None
            cycle = [None, "running", "completions", "pending", "stats"]
            current_idx = cycle.index(self._sort_table) if self._sort_table in cycle else 0
            self._sort_table = cycle[(current_idx - 1) % len(cycle)]
            self._sort_column = 0
            self._sort_ascending = True
            self._force_refresh = True
            return True
        if key in "123" and self._sort_table is not None:
            col = int(key) - 1  # Convert to 0-indexed
            # Completions and pending tables only have limited columns
            if self._sort_table == "completions" and col > 2:
                return True
            if self._sort_table == "pending" and col > 1:
                return True
            if col == self._sort_column:
                self._sort_ascending = not self._sort_ascending
            else:
                self._sort_column = col
                self._sort_ascending = True
            self._force_refresh = True
            return True
        if key == "4" and self._sort_table in ("running", "stats"):
            col = 3
            if col == self._sort_column:
                self._sort_ascending = not self._sort_ascending
            else:
                self._sort_column = col
                self._sort_ascending = True
            self._force_refresh = True
            return True
        return False

    def _handle_log_navigation_key(self, key: str) -> bool:
        """Handle log navigation keys ([, ], {, }). Returns True if handled."""
        if key == "[":  # Older log (1 step)
            self._refresh_log_list()
            if self._current_log_index < len(self._available_logs) - 1:
                self._current_log_index += 1
                self._force_refresh = True
            return True
        if key == "]":  # Newer log (1 step)
            if self._current_log_index > 0:
                self._current_log_index -= 1
                self._force_refresh = True
            return True
        if key == "{":  # Older log (5 steps)
            self._refresh_log_list()
            max_idx = len(self._available_logs) - 1
            self._current_log_index = min(max_idx, self._current_log_index + 5)
            self._force_refresh = True
            return True
        if key == "}":  # Newer log (5 steps)
            self._current_log_index = max(0, self._current_log_index - 5)
            self._force_refresh = True
            return True
        return False

    def _handle_filter_key(self, key: str) -> bool:
        """Handle keypress in filter input mode."""
        if key == "\x1b":  # Escape
            self._filter_mode = False
            self._filter_input = ""
            self._force_refresh = True
        elif key == "\r" or key == "\n":  # Enter
            self._filter_mode = False
            self._filter_text = self._filter_input if self._filter_input else None
            self._filter_index = 0
            self._force_refresh = True
        elif key == "\x7f" or key == "\b":  # Backspace
            self._filter_input = self._filter_input[:-1]
            self._force_refresh = True
        elif key.isprintable():
            self._filter_input += key
            self._force_refresh = True
        return False

    def _handle_job_selection_key(self, key: str, num_jobs: int) -> bool:
        """Handle keypress in job selection mode. Returns True if should quit."""
        if key == "\x1b":  # Escape - exit selection mode
            self._job_selection_mode = False
            self._log_source = None
            self._selected_job_index = 0
            self._selected_completion_index = 0
            self._log_scroll_offset = 0
            self._cached_log_path = None
            self._cached_log_lines = []
            self._force_refresh = True
            return False

        # Tab - cycle between running and completions log sources
        if key == "\t":
            if self._log_source == "running":
                self._log_source = "completions"
            else:
                self._log_source = "running"
            self._log_scroll_offset = 0  # Reset scroll on source change
            self._force_refresh = True
            return False

        # Job navigation: Ctrl+p (up) or Ctrl+n (down)
        # Use appropriate index based on log source
        if key == "\x10":  # Ctrl+p - previous job
            if self._log_source == "completions":
                self._selected_completion_index = max(0, self._selected_completion_index - 1)
            else:
                self._selected_job_index = max(0, self._selected_job_index - 1)
            self._log_scroll_offset = 0  # Reset scroll on job change
            self._force_refresh = True
            return False

        if key == "\x0e":  # Ctrl+n - next job
            if self._log_source == "completions":
                self._selected_completion_index = min(
                    num_jobs - 1, self._selected_completion_index + 1
                )
            else:
                self._selected_job_index = min(num_jobs - 1, self._selected_job_index + 1)
            self._log_scroll_offset = 0
            self._force_refresh = True
            return False

        # Log scrolling: Ctrl+u (up) or Ctrl+d (down)
        if key == "\x15":  # Ctrl+u - scroll log up (show older)
            self._log_scroll_offset += 10
            self._force_refresh = True
            return False

        if key == "\x04":  # Ctrl+d - scroll log down (show newer)
            self._log_scroll_offset = max(0, self._log_scroll_offset - 10)
            self._force_refresh = True
            return False

        # Jump to top/bottom of log
        if key == "g":  # Jump to top of log
            self._log_scroll_offset = max(0, len(self._cached_log_lines) - 10)
            self._force_refresh = True
            return False

        if key == "G":  # Jump to bottom (latest)
            self._log_scroll_offset = 0
            self._force_refresh = True
            return False

        return False

    def _make_easter_egg_panel(self) -> Panel:
        """Create the Fulcrum Genomics easter egg panel."""
        from rich.align import Align

        # Try to load and display the logo image
        if FG_LOGO_PATH.exists():
            try:
                from PIL import Image  # type: ignore[import-not-found]
                from rich_pixels import Pixels  # type: ignore[import-not-found]

                # Load and resize image to fit terminal
                img = Image.open(FG_LOGO_PATH)

                # Calculate size based on terminal
                # Each character cell is roughly 2x1 aspect ratio (taller than wide)
                # rich-pixels uses half-block characters, so 2 pixels per character height
                term_width = self.console.width or 80
                term_height = self.console.height or 24

                # Target size: use most of the terminal, leave minimal border
                max_char_width = term_width - 4  # Leave room for panel border
                max_char_height = term_height - 5  # Leave room for border + footer

                # Convert to pixel dimensions
                # Width: 1 char = ~1 pixel width in rich-pixels
                # Height: 1 char = 2 pixel rows (half-block characters)
                max_pixel_width = max_char_width
                max_pixel_height = max_char_height * 2

                # Maintain aspect ratio
                img_ratio = img.width / img.height
                target_ratio = max_pixel_width / max_pixel_height

                if target_ratio > img_ratio:
                    # Terminal is wider than image - fit to height
                    new_height = max_pixel_height
                    new_width = int(new_height * img_ratio)
                else:
                    # Terminal is taller than image - fit to width
                    new_width = max_pixel_width
                    new_height = int(new_width / img_ratio)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                pixels = Pixels.from_image(img)

                centered = Align.center(pixels, vertical="middle")

                return Panel(
                    centered,
                    border_style=FG_BLUE,
                    subtitle="[dim]Press any key to return[/dim]",
                    subtitle_align="center",
                )
            except ImportError:
                pass  # Fall through to text fallback
            except Exception:
                pass  # Fall through to text fallback

        # Fallback: simple text logo
        fallback = Text()
        fallback.append("\n\n")
        fallback.append("  FULCRUM GENOMICS  ", style=f"bold {FG_BLUE}")
        fallback.append("\n\n")
        fallback.append("  Press any key to return...", style="dim")

        return Panel(
            Align.center(fallback, vertical="middle"),
            border_style=FG_BLUE,
            style="on black",
        )

    def _make_help_panel(self) -> Panel:
        """Create the help overlay panel."""
        help_text = Table(show_header=False, box=None, padding=(0, 2))
        help_text.add_column("Key", style="bold cyan")
        help_text.add_column("Action")

        help_text.add_row("", "[bold]General[/bold]")
        help_text.add_row("q", "Quit")
        help_text.add_row("?", "Toggle this help")
        help_text.add_row("p", "Pause/resume auto-refresh")
        help_text.add_row("e", "Toggle time estimation")
        help_text.add_row("w", "Toggle wildcard conditioning")
        help_text.add_row("r", "Force refresh")
        help_text.add_row("Ctrl+r", "Hard refresh (reload historical data)")
        help_text.add_row("", "")
        help_text.add_row("", "[bold]Refresh Rate (vim-style)[/bold]")
        help_text.add_row("h", "Decrease by 5s (faster)")
        help_text.add_row("j", "Decrease by 0.5s (faster)")
        help_text.add_row("k", "Increase by 0.5s (slower)")
        help_text.add_row("l", "Increase by 5s (slower)")
        help_text.add_row("0", f"Reset to default ({DEFAULT_REFRESH_RATE}s)")
        help_text.add_row("G", f"Set to minimum ({MIN_REFRESH_RATE}s, fastest)")
        help_text.add_row("", "")
        help_text.add_row("", "[bold]Layout & Filter[/bold]")
        help_text.add_row("Tab", "Cycle layout (full/compact/minimal)")
        help_text.add_row("/", "Filter rules by name")
        help_text.add_row("n / N", "Next/previous filter match")
        help_text.add_row("Esc", "Clear filter, return to latest log")
        help_text.add_row("", "")
        help_text.add_row("", "[bold]Log Navigation[/bold]")
        help_text.add_row("[ / ]", "View older/newer log (1 step)")
        help_text.add_row("{ / }", "View older/newer log (5 steps)")
        help_text.add_row("", "")
        help_text.add_row("", "[bold]Table Sorting[/bold]")
        help_text.add_row("s / S", "Cycle sort table (forward/backward)")
        help_text.add_row("1-4", "Sort by column (press again to reverse)")
        help_text.add_row("", "")
        help_text.add_row("", "[bold]Job Log Viewer[/bold]")
        help_text.add_row("Enter", "Toggle job log view")
        help_text.add_row("Ctrl+p / Ctrl+n", "Select prev/next job")
        help_text.add_row("Ctrl+u / Ctrl+d", "Scroll log up/down")
        help_text.add_row("g / G", "Jump to top/bottom of log")

        return Panel(
            help_text,
            title="[bold]Keyboard Shortcuts[/bold]",
            subtitle="Press any key to close",
            border_style="cyan",
        )

    def _make_header(self, progress: WorkflowProgress) -> Panel:
        """Create the header panel with workflow path and status."""
        status_styles = {
            WorkflowStatus.RUNNING: "bold green",
            WorkflowStatus.COMPLETED: "bold blue",
            WorkflowStatus.FAILED: "bold red",
            WorkflowStatus.INCOMPLETE: "bold yellow",
            WorkflowStatus.UNKNOWN: "bold yellow",
        }
        style = status_styles.get(progress.status, "bold white")

        header_text = Text()
        header_text.append("FULCRUM GENOMICS", style=f"bold {FG_BLUE}")
        header_text.append(" │ ", style="dim")
        header_text.append("Snakemake Monitor", style="bold white")
        header_text.append("  │  ", style="dim")
        header_text.append(str(self.workflow_dir), style="dim")
        header_text.append("  │  Status: ")
        header_text.append(progress.status.value.upper(), style=style)

        if progress.elapsed_seconds is not None:
            header_text.append("  │  Elapsed: ")
            header_text.append(format_duration(progress.elapsed_seconds), style=FG_BLUE)

        if self._paused:
            header_text.append("  │  ")
            header_text.append("PAUSED", style="bold yellow")

        return Panel(header_text, style="white on grey23", border_style=FG_BLUE, height=3)

    def _make_progress_bar(self, progress: WorkflowProgress, width: int = 40) -> Text:
        """Create a colored progress bar showing succeeded/failed portions."""
        total = max(1, progress.total_jobs)
        succeeded = progress.completed_jobs
        failed = progress.failed_jobs

        # Calculate widths for each segment
        succeeded_width = int((succeeded / total) * width)
        failed_width = int((failed / total) * width)
        remaining_width = width - succeeded_width - failed_width

        # Build the bar with colored segments
        bar = Text()
        bar.append("█" * succeeded_width, style="green")
        bar.append("█" * failed_width, style="red")
        if progress.status == WorkflowStatus.INCOMPLETE:
            bar.append("░" * remaining_width, style="yellow")  # Incomplete = yellow
        else:
            bar.append("░" * remaining_width, style="dim")

        return bar

    def _make_progress_panel(
        self,
        progress: WorkflowProgress,
        estimate: TimeEstimate | None,
    ) -> Panel:
        """Create the progress bar panel."""
        total = max(1, progress.total_jobs)
        completed = progress.completed_jobs + progress.failed_jobs
        percent = (completed / total) * 100

        # Calculate bar width based on console width
        # Reserve space for: "Progress " (9) + " XX.X% " (7) + "(XXX/XXX jobs)" (~15) + borders (~4)
        console_width = self.console.width or 80
        bar_width = max(20, console_width - 40)

        # Create colored progress bar
        progress_bar = self._make_progress_bar(progress, width=bar_width)

        # Progress text line
        progress_line = Text()
        progress_line.append("Progress ", style=f"bold {FG_BLUE}")
        progress_line.append(progress_bar)
        progress_line.append(f" {percent:5.1f}% ", style="bold")
        progress_line.append(f"({completed}/{total} jobs)", style="dim")

        # ETA text - handle different workflow states
        eta_parts = []
        if progress.status == WorkflowStatus.FAILED:
            eta_parts.append("[bold red]FAILED[/bold red]")
            if progress.failed_jobs > 0:
                eta_parts.append(f"[dim]({progress.failed_jobs} job(s) failed)[/dim]")
        elif progress.status == WorkflowStatus.INCOMPLETE:
            eta_parts.append("[bold yellow]INCOMPLETE[/bold yellow]")
            if progress.incomplete_jobs_list:
                eta_parts.append(
                    f"[dim]({len(progress.incomplete_jobs_list)} job(s) were in progress)[/dim]"
                )
        elif progress.status == WorkflowStatus.COMPLETED:
            eta_parts.append("[bold blue]Complete[/bold blue]")
        elif estimate is not None:
            eta_parts.append(f"ETA: {estimate.format_eta()}")

            if estimate.seconds_remaining < float("inf") and estimate.seconds_remaining > 0:
                completion_time = datetime.now().timestamp() + estimate.seconds_remaining
                completion_str = datetime.fromtimestamp(completion_time).strftime("%H:%M:%S")
                eta_parts.append(f"(completion: {completion_str})")

            # Show estimation method for transparency
            eta_parts.append(f"[dim][{estimate.method}][/dim]")
        elif not self.use_estimation:
            eta_parts.append("[dim]ETA: disabled[/dim]")

        eta_text = Text.from_markup("  ".join(eta_parts)) if eta_parts else Text("")

        # Legend for the progress bar when there are failures
        legend = Text()
        if progress.failed_jobs > 0:
            legend.append("  (", style="dim")
            legend.append("█", style="green")
            legend.append(f"={progress.completed_jobs} succeeded  ", style="dim")
            legend.append("█", style="red")
            legend.append(f"={progress.failed_jobs} failed", style="dim")
            legend.append(")", style="dim")

        # Border color based on status (use FG colors for normal states)
        border_colors = {
            WorkflowStatus.RUNNING: FG_BLUE,
            WorkflowStatus.COMPLETED: FG_GREEN,
            WorkflowStatus.FAILED: "red",
            WorkflowStatus.INCOMPLETE: "yellow",
            WorkflowStatus.UNKNOWN: "yellow",
        }
        border_style = border_colors.get(progress.status, FG_BLUE)

        # Combine progress line with legend if present
        if progress.failed_jobs > 0:
            full_progress = Text()
            full_progress.append(progress_line)
            full_progress.append(legend)
            return Panel(
                Group(full_progress, eta_text),
                title="Progress",
                border_style=border_style,
            )

        return Panel(
            Group(progress_line, eta_text),
            title="Progress",
            border_style=border_style,
        )

    def _filter_jobs(self, jobs: list[JobInfo]) -> list[JobInfo]:
        """Filter jobs by rule name if filter is active."""
        if not self._filter_text:
            return jobs

        filtered = [j for j in jobs if self._filter_text.lower() in j.rule.lower()]

        # Update filter matches for n/N navigation
        self._filter_matches = list({j.rule for j in filtered})

        return filtered

    def _sort_indicator(self, table_name: str, col: int) -> str:
        """Get sort indicator for a column header."""
        if self._sort_table != table_name or self._sort_column != col:
            return ""
        return " ▲" if self._sort_ascending else " ▼"

    def _get_tool_progress(self, job: JobInfo) -> ToolProgress | None:
        """
        Get tool-specific progress for a running job.

        Args:
            job: The running job to check.

        Returns:
            ToolProgress if parseable, None otherwise.
        """
        # Find the log file for this job
        log_path = find_rule_log(job.rule, job.job_id, self.workflow_dir, job.wildcards)
        if log_path is None:
            return None

        # Try to parse progress from the log
        return parse_tool_progress(job.rule, log_path)

    def _build_running_job_data(
        self, jobs: list[JobInfo]
    ) -> list[tuple[JobInfo, float | None, float | None, float | None, ToolProgress | None]]:
        """Build sortable data for running jobs."""
        job_data: list[
            tuple[JobInfo, float | None, float | None, float | None, ToolProgress | None]
        ] = []
        for job in jobs:
            elapsed = job.elapsed
            remaining: float | None = None
            tool_progress: ToolProgress | None = None

            if self._estimator is not None and elapsed is not None:
                # Use thread-aware ETA when thread info is available
                expected, _ = self._estimator.get_estimate_for_job(
                    rule=job.rule,
                    threads=job.threads,
                )
                remaining = max(0, expected - elapsed)

            # Try to get tool-specific progress
            tool_progress = self._get_tool_progress(job)

            # If we have tool progress with percentage, use it to improve ETA
            if tool_progress is not None and tool_progress.percent_complete is not None:
                if elapsed is not None and tool_progress.percent_complete > 0:
                    # Estimate remaining time based on progress
                    pct = tool_progress.percent_complete / 100.0
                    tool_remaining = elapsed * (1 - pct) / pct if pct > 0 else None
                    # Prefer tool-based estimate if available
                    if tool_remaining is not None:
                        remaining = tool_remaining

            job_data.append((job, elapsed, remaining, job.start_time, tool_progress))
        return job_data

    def _sort_running_job_data(
        self,
        job_data: list[
            tuple[JobInfo, float | None, float | None, float | None, ToolProgress | None]
        ],
    ) -> list[tuple[JobInfo, float | None, float | None, float | None, ToolProgress | None]]:
        """Sort running job data based on current sort settings."""
        if not job_data:
            return job_data
        sort_keys = {
            0: lambda x: x[0].rule.lower(),
            1: lambda x: x[3] or 0,
            2: lambda x: x[1] or 0,
            3: lambda x: x[2] if x[2] is not None else float("inf"),
        }
        key_fn = sort_keys.get(self._sort_column, sort_keys[0])
        return sorted(job_data, key=key_fn, reverse=not self._sort_ascending)

    def _make_running_table(self, progress: WorkflowProgress) -> Panel:  # noqa: C901
        """Create the currently running jobs table."""
        is_sorting = self._sort_table == "running"
        header_style = "bold magenta on dark_blue" if is_sorting else "bold magenta"

        table = Table(expand=True, show_header=True, header_style=header_style)
        table.add_column(f"Rule{self._sort_indicator('running', 0)}", style="cyan", no_wrap=True)
        table.add_column("Thr", justify="right", style="dim")
        table.add_column(f"Started{self._sort_indicator('running', 1)}", justify="right")
        table.add_column(f"Elapsed{self._sort_indicator('running', 2)}", justify="right")
        table.add_column("Progress", justify="right")
        table.add_column(f"Est. Remaining{self._sort_indicator('running', 3)}", justify="right")

        jobs = self._filter_jobs(progress.running_jobs)
        job_data = self._build_running_job_data(jobs)

        if is_sorting:
            job_data = self._sort_running_job_data(job_data)

        for idx, (job, elapsed, remaining, _start, tool_progress) in enumerate(job_data[:10]):
            elapsed_str = format_duration(elapsed) if elapsed is not None else "?"
            remaining_str = f"~{format_duration(remaining)}" if remaining is not None else "?"
            started_str = "?"
            if job.start_time is not None:
                started_str = datetime.fromtimestamp(job.start_time).strftime("%H:%M:%S")

            # Format tool progress
            progress_str = ""
            if tool_progress is not None:
                if tool_progress.percent_complete is not None:
                    progress_str = f"{tool_progress.percent_str}"
                else:
                    # Show items processed if no percentage
                    progress_str = f"{tool_progress.items_processed:,} {tool_progress.unit}"

            rule_style = "cyan"
            # Highlight selected job in selection mode (only when viewing running jobs)
            is_selecting_running = self._job_selection_mode and self._log_source == "running"
            if is_selecting_running and idx == self._selected_job_index:
                rule_style = "bold cyan on dark_blue"
            elif self._filter_matches and self._filter_index < len(self._filter_matches):
                if job.rule == self._filter_matches[self._filter_index]:
                    rule_style = "bold cyan on dark_blue"

            threads_str = str(job.threads) if job.threads is not None else "-"
            table.add_row(
                Text(job.rule, style=rule_style),
                threads_str,
                started_str,
                elapsed_str,
                Text(progress_str, style="green") if progress_str else Text("-", style="dim"),
                remaining_str,
            )

        if not jobs:
            msg = f"[dim]No jobs matching '{self._filter_text}'[/dim]" if self._filter_text else ""
            msg = msg or "[dim]No jobs currently running[/dim]"
            table.add_row(msg, "", "", "", "", "")

        is_selecting_running = self._job_selection_mode and self._log_source == "running"
        title = f"Currently Running ({len(progress.running_jobs)} jobs)"
        if is_selecting_running:
            title += " [bold cyan]◀ select job[/bold cyan]"
        elif is_sorting:
            title += " [bold cyan]◀ sorting[/bold cyan]"
        if self._filter_text:
            title += f" [dim]filter: {self._filter_text}[/dim]"
        border = "cyan" if is_selecting_running else (f"bold {FG_BLUE}" if is_sorting else FG_BLUE)
        return Panel(table, title=title, border_style=border, padding=0)

    def _make_completions_table(self, progress: WorkflowProgress) -> Panel:
        """Create the recent completions table."""
        is_sorting = self._sort_table == "completions"
        header_style = "bold green on dark_blue" if is_sorting else "bold green"

        table = Table(expand=True, show_header=True, header_style=header_style)
        ind = self._sort_indicator
        table.add_column(f"Rule{ind('completions', 0)}", no_wrap=True)
        table.add_column(f"Duration{ind('completions', 1)}", justify="right")
        table.add_column(f"Completed{ind('completions', 2)}", justify="right")

        # Merge successful completions and failed jobs for a unified view
        failed_job_ids = {id(job) for job in progress.failed_jobs_list}
        all_jobs = list(progress.recent_completions) + list(progress.failed_jobs_list)

        # Sort by end_time (most recent first) by default
        all_jobs.sort(key=lambda j: j.end_time or 0, reverse=True)

        jobs = self._filter_jobs(all_jobs)

        # Sort if this table is active
        if is_sorting and jobs:
            sort_keys = {
                0: lambda j: j.rule.lower(),
                1: lambda j: j.duration or 0,
                2: lambda j: j.end_time or 0,
            }
            key_fn = sort_keys.get(self._sort_column, sort_keys[2])
            jobs = sorted(jobs, key=key_fn, reverse=not self._sort_ascending)

        # Check if we're in completions selection mode
        is_selecting = self._job_selection_mode and self._log_source == "completions"

        for idx, job in enumerate(jobs[:8]):  # Limit to 8 rows
            duration = job.duration
            duration_str = format_duration(duration) if duration is not None else "?"

            completed_str = "?"
            if job.end_time is not None:
                completed_str = datetime.fromtimestamp(job.end_time).strftime("%H:%M:%S")

            # Color based on success/failure status
            is_failed = id(job) in failed_job_ids
            rule_style = "red" if is_failed else "cyan"
            time_style = "red" if is_failed else "green"

            # Highlight selected job in selection mode
            if is_selecting and idx == self._selected_completion_index:
                rule_style = "bold cyan on dark_blue" if not is_failed else "bold red on dark_blue"

            table.add_row(
                Text(job.rule, style=rule_style),
                duration_str,
                f"[{time_style}]{completed_str}[/{time_style}]",
            )

        if not jobs:
            msg = f"[dim]No jobs matching '{self._filter_text}'[/dim]" if self._filter_text else ""
            msg = msg or "[dim]No completed jobs yet[/dim]"
            table.add_row(msg, "", "")

        title = "Recent Completions"
        if is_selecting:
            title += " [bold cyan]◀ select job[/bold cyan]"
        elif is_sorting:
            title += " [bold cyan]◀ sorting[/bold cyan]"
        border = "cyan" if is_selecting else (f"bold {FG_BLUE}" if is_sorting else FG_BLUE)
        return Panel(table, title=title, border_style=border, padding=0)

    def _make_summary_footer(self, progress: WorkflowProgress) -> Panel:
        """Create the job status summary as a one-line footer panel."""
        succeeded = progress.completed_jobs
        failed = progress.failed_jobs
        running = len(progress.running_jobs)
        incomplete = len(progress.incomplete_jobs_list)
        pending = progress.pending_jobs

        summary = Text()
        summary.append("Jobs: ", style="dim")
        summary.append(f"{succeeded}", style="green")
        summary.append(" succeeded", style="dim")
        summary.append("  │  ", style="dim")
        summary.append(f"{failed}", style="red" if failed > 0 else "dim")
        summary.append(" failed", style="dim")
        summary.append("  │  ", style="dim")
        summary.append(f"{running}", style="cyan" if running > 0 else "dim")
        summary.append(" running", style="dim")
        # Show incomplete count if there are incomplete jobs
        if incomplete > 0:
            summary.append("  │  ", style="dim")
            summary.append(f"{incomplete}", style="yellow")
            summary.append(" incomplete", style="dim")
        summary.append("  │  ", style="dim")
        summary.append(f"{pending}", style="yellow" if pending > 0 else "dim")
        summary.append(" pending", style="dim")

        border_style = "red" if failed > 0 else FG_BLUE
        return Panel(summary, border_style=border_style, padding=(0, 1))

    def _get_inferred_pending_rules(self, progress: WorkflowProgress) -> dict[str, int] | None:
        """Get inferred pending rules from completions and historical data."""
        if not self._estimator or not progress.recent_completions:
            return None

        # Count completed jobs by rule
        completed_by_rule: dict[str, int] = {}
        for job in progress.recent_completions:
            completed_by_rule[job.rule] = completed_by_rule.get(job.rule, 0) + 1

        # Also include rule stats from historical data
        if self._estimator.rule_stats:
            for rule, stats in self._estimator.rule_stats.items():
                if rule not in completed_by_rule:
                    completed_by_rule[rule] = stats.count

        return self._estimator._infer_pending_rules(completed_by_rule, progress.pending_jobs)

    def _read_log_tail(self, log_path: Path, max_lines: int = 500) -> list[str]:
        """
        Read the last N lines of a log file efficiently.

        Args:
            log_path: Path to the log file.
            max_lines: Maximum number of lines to read.

        Returns:
            List of lines (most recent at end).
        """
        try:
            # Check if cache is still valid
            mtime = log_path.stat().st_mtime
            if (
                self._cached_log_path == log_path
                and self._cached_log_mtime == mtime
                and self._cached_log_lines
            ):
                return self._cached_log_lines

            # Read file and take last N lines
            content = log_path.read_text(errors="ignore")
            lines = content.splitlines()

            # Take last max_lines
            result = lines[-max_lines:] if len(lines) > max_lines else lines

            # Update cache
            self._cached_log_path = log_path
            self._cached_log_mtime = mtime
            self._cached_log_lines = result

            return result
        except OSError:
            return ["[Error reading log file]"]

    def _get_completions_list(self, progress: WorkflowProgress) -> tuple[list[JobInfo], set[int]]:
        """Get merged list of completed and failed jobs with same order as table.

        Applies the same filtering and sorting as _make_completions_table() to ensure
        the selected index matches between the table display and log panel.

        Returns:
            Tuple of (jobs_list, failed_job_ids_set)
        """
        failed_job_ids = {id(job) for job in progress.failed_jobs_list}
        all_jobs = list(progress.recent_completions) + list(progress.failed_jobs_list)

        # Sort by end_time (most recent first) by default - matches table
        all_jobs.sort(key=lambda j: j.end_time or 0, reverse=True)

        # Apply filtering - must match _make_completions_table
        jobs = self._filter_jobs(all_jobs)

        # Apply custom sorting if completions table is being sorted
        is_sorting = self._sort_table == "completions"
        if is_sorting and jobs:
            sort_keys = {
                0: lambda j: j.rule.lower(),
                1: lambda j: j.duration or 0,
                2: lambda j: j.end_time or 0,
            }
            key_fn = sort_keys.get(self._sort_column, sort_keys[2])
            jobs = sorted(jobs, key=key_fn, reverse=not self._sort_ascending)

        return jobs, failed_job_ids

    def _get_running_jobs_list(self, progress: WorkflowProgress) -> list[JobInfo]:
        """Get running jobs list with same order as table.

        Applies the same filtering and sorting as _make_running_table() to ensure
        the selected index matches between the table display and log panel.

        Returns:
            List of running jobs in display order.
        """
        jobs = self._filter_jobs(progress.running_jobs)

        # Apply custom sorting if running table is being sorted
        is_sorting = self._sort_table == "running"
        if is_sorting and jobs:
            # Build job data tuples for sorting (same as _make_running_table)
            job_data = self._build_running_job_data(jobs)
            job_data = self._sort_running_job_data(job_data)
            # Extract just the jobs from the sorted tuples
            jobs = [jd[0] for jd in job_data]

        return jobs

    def _make_job_log_panel(self, progress: WorkflowProgress) -> Panel:
        """Create the job log panel showing selected job's log content."""
        # Build job lists for both sources - use same ordering as tables
        running_jobs = self._get_running_jobs_list(progress)
        completions, failed_ids = self._get_completions_list(progress)

        # Determine which source to use and get selected job
        selected_job: JobInfo | None = None
        job_status: str = ""  # "", "completed", or "failed"

        if self._log_source == "running":
            if running_jobs:
                self._selected_job_index = min(
                    self._selected_job_index,
                    len(running_jobs) - 1,
                )
                selected_job = running_jobs[self._selected_job_index]
        elif self._log_source == "completions":
            if completions:
                self._selected_completion_index = min(
                    self._selected_completion_index,
                    len(completions) - 1,
                )
                selected_job = completions[self._selected_completion_index]
                job_status = "failed" if id(selected_job) in failed_ids else "completed"

        # Handle no jobs available
        if selected_job is None:
            source_name = "running jobs" if self._log_source == "running" else "completed jobs"
            return Panel(
                f"[dim]No {source_name}[/dim]",
                title="Job Log",
                subtitle="[dim]Tab switch source | Esc exit[/dim]",
                border_style=FG_BLUE,
            )

        # Find log file using existing plugin infrastructure
        log_path = find_rule_log(
            selected_job.rule,
            selected_job.job_id,
            self.workflow_dir,
            selected_job.wildcards,
        )

        if log_path is None:
            status_suffix = f" [{job_status}]" if job_status else ""
            return Panel(
                f"[dim]No log file found for {selected_job.rule}[/dim]",
                title=f"Job Log: {selected_job.rule}{status_suffix}",
                subtitle="[dim]Tab switch source | Esc exit[/dim]",
                border_style=FG_BLUE,
            )

        # Read log content
        lines = self._read_log_tail(log_path, max_lines=500)

        if not lines:
            status_suffix = f" [{job_status}]" if job_status else ""
            return Panel(
                "[dim]Log file is empty[/dim]",
                title=f"Job Log: {selected_job.rule}{status_suffix}",
                subtitle="[dim]Tab switch source | Esc exit[/dim]",
                border_style=FG_BLUE,
            )

        # Calculate visible window based on console height
        # Panel takes ~5 lines for border, title, and padding
        panel_height = (self.console.height or 24) // 3 - 5
        visible_lines = max(5, panel_height)

        # Apply scroll offset (0 = show latest, positive = show older)
        total_lines = len(lines)
        end_index = total_lines - self._log_scroll_offset
        start_index = max(0, end_index - visible_lines)
        end_index = max(start_index + 1, end_index)

        # Clamp scroll offset
        max_offset = max(0, total_lines - visible_lines)
        self._log_scroll_offset = min(self._log_scroll_offset, max_offset)

        visible_content = lines[start_index:end_index]

        # Build display
        content = Text()
        for i, line in enumerate(visible_content):
            # Truncate long lines
            display_line = line[:117] + "..." if len(line) > 120 else line
            # Highlight recent lines
            style = "white" if i >= len(visible_content) - 3 else "dim"
            content.append(display_line + "\n", style=style)

        # Build title with scroll indicator and status
        scroll_indicator = ""
        if self._log_scroll_offset > 0:
            scroll_indicator = f" [line {start_index + 1}-{end_index}/{total_lines}]"
        elif total_lines > visible_lines:
            scroll_indicator = f" [latest {visible_lines}/{total_lines}]"

        status_suffix = f" [{job_status}]" if job_status else ""
        title = f"Job Log: {selected_job.rule}{status_suffix}{scroll_indicator}"
        subtitle = "[dim]Ctrl+u/d scroll | g/G top/bottom | Tab switch | Esc exit[/dim]"

        return Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style="cyan",
        )

    def _make_pending_jobs_panel(self, progress: WorkflowProgress) -> Panel:
        """Create the pending jobs panel showing estimated pending jobs by rule."""
        pending_count = progress.pending_jobs
        is_sorting = self._sort_table == "pending"
        title_suffix = " [bold cyan]◀ sorting[/bold cyan]" if is_sorting else ""
        border = f"bold {FG_BLUE}" if is_sorting else FG_BLUE

        if pending_count <= 0:
            return Panel(
                "[dim]No pending jobs[/dim]",
                title="Pending Jobs" + title_suffix,
                border_style=border,
            )

        pending_rules = self._get_inferred_pending_rules(progress)
        if not pending_rules:
            msg = f"[yellow]{pending_count}[/yellow] [dim]jobs pending[/dim]"
            return Panel(msg, title="Pending Jobs" + title_suffix, border_style=border)

        table = Table(expand=True, show_header=True, header_style=f"bold {FG_BLUE}")
        table.add_column(f"Rule{self._sort_indicator('pending', 0)}", style="yellow", no_wrap=True)
        table.add_column(
            f"Est. Count{self._sort_indicator('pending', 1)}", justify="right", style="dim"
        )

        # Sort based on current sort settings or default to count descending
        rules_list = list(pending_rules.items())
        if is_sorting and self._sort_column == 0:
            rules_list.sort(key=lambda x: x[0].lower(), reverse=not self._sort_ascending)
        elif is_sorting:
            rules_list.sort(key=lambda x: x[1], reverse=not self._sort_ascending)
        else:
            rules_list.sort(key=lambda x: x[1], reverse=True)

        for rule, count in rules_list[:10]:
            table.add_row(rule, str(count))

        if len(rules_list) > 10:
            table.add_row(f"[dim]... and {len(rules_list) - 10} more rules[/dim]", "")

        return Panel(
            table,
            title=f"Pending Jobs (~{pending_count})" + title_suffix,
            border_style=border,
            padding=0,
        )

    def _make_failed_jobs_panel(self, progress: WorkflowProgress) -> Panel:
        """Create the failed jobs list panel."""
        if not progress.failed_jobs_list:
            return Panel(
                "[dim]No failed jobs[/dim]",
                title="Failed Jobs",
                border_style="dim",
            )

        table = Table(expand=True, show_header=True, header_style="bold red")
        table.add_column("Rule", style="red", no_wrap=True)
        table.add_column("Job ID", justify="right", style="dim")

        for job in progress.failed_jobs_list[:8]:  # Limit to 8 rows
            job_id_str = job.job_id if job.job_id else "-"
            table.add_row(job.rule, job_id_str)

        more_count = len(progress.failed_jobs_list) - 8
        if more_count > 0:
            table.add_row(f"[dim]... and {more_count} more[/dim]", "")

        return Panel(
            table,
            title=f"Failed Jobs ({len(progress.failed_jobs_list)})",
            border_style="red",
            padding=0,
        )

    def _make_incomplete_jobs_panel(self, progress: WorkflowProgress) -> Panel:
        """Create the incomplete jobs list panel showing files that were in progress."""
        if not progress.incomplete_jobs_list:
            return Panel(
                "[dim]No incomplete jobs[/dim]",
                title="Incomplete Jobs",
                border_style="dim",
            )

        table = Table(expand=True, show_header=True, header_style="bold yellow")
        table.add_column("Output File", style="yellow", no_wrap=False)

        for job in progress.incomplete_jobs_list[:8]:  # Limit to 8 rows
            if job.output_file:
                # Show relative path if possible, otherwise full path
                try:
                    rel_path = job.output_file.relative_to(progress.workflow_dir)
                    table.add_row(str(rel_path))
                except ValueError:
                    table.add_row(str(job.output_file))
            else:
                table.add_row("[dim]unknown[/dim]")

        more_count = len(progress.incomplete_jobs_list) - 8
        if more_count > 0:
            table.add_row(f"[dim]... and {more_count} more[/dim]")

        return Panel(
            table,
            title=f"Incomplete Jobs ({len(progress.incomplete_jobs_list)})",
            border_style="yellow",
            padding=0,
        )

    def _parse_stats_from_logs(self, cutoff: float) -> dict[str, RuleTimingStats]:
        """Parse rule stats from log files created before the cutoff time."""
        from snakesee.parser import parse_completed_jobs_from_log

        stats_dict: dict[str, RuleTimingStats] = {}
        for log in self._available_logs:
            try:
                if log.stat().st_ctime >= cutoff:
                    continue
                for job in parse_completed_jobs_from_log(log):
                    if job.duration is not None:
                        if job.rule not in stats_dict:
                            stats_dict[job.rule] = RuleTimingStats(rule=job.rule)
                        stats_dict[job.rule].durations.append(job.duration)
            except OSError:
                continue
        return stats_dict

    def _get_filtered_stats(self) -> list[RuleTimingStats]:
        """Get rule stats filtered by cutoff time if viewing historical log."""
        from snakesee.parser import parse_metadata_files

        if self._cutoff_time is None:
            # Latest log: use all stats from estimator
            if self._estimator and self._estimator.rule_stats:
                return list(self._estimator.rule_stats.values())
            return []

        # Historical log: rebuild stats from metadata, filtering by cutoff time
        metadata_dir = self.workflow_dir / ".snakemake" / "metadata"
        stats_dict: dict[str, RuleTimingStats] = {}
        for job in parse_metadata_files(metadata_dir):
            if job.duration is not None and job.end_time is not None:
                if job.end_time < self._cutoff_time:
                    if job.rule not in stats_dict:
                        stats_dict[job.rule] = RuleTimingStats(rule=job.rule)
                    stats_dict[job.rule].durations.append(job.duration)

        # If no metadata found, parse stats from log files up to the cutoff
        if not stats_dict:
            stats_dict = self._parse_stats_from_logs(self._cutoff_time)

        return list(stats_dict.values())

    def _make_stats_panel(self) -> Panel:
        """Create the rule statistics panel."""
        is_sorting = self._sort_table == "stats"

        # Check if estimation is disabled
        if not self.use_estimation:
            return Panel(
                "[dim]Estimation disabled[/dim]",
                title="Rule Statistics (Historical)",
                border_style=FG_BLUE,
            )

        stats_list = self._get_filtered_stats()

        if not stats_list:
            return Panel(
                "[dim]No historical data available[/dim]",
                title="Rule Statistics (Historical)",
                border_style=FG_BLUE,
            )

        header_style = "bold yellow on dark_blue" if is_sorting else "bold yellow"
        table = Table(expand=True, show_header=True, header_style=header_style)
        table.add_column(f"Rule{self._sort_indicator('stats', 0)}", style="cyan", no_wrap=True)
        table.add_column("Thr", justify="right", style="dim")
        table.add_column(f"Count{self._sort_indicator('stats', 1)}", justify="right")
        table.add_column(f"Avg Time{self._sort_indicator('stats', 2)}", justify="right")
        table.add_column(f"Std Dev{self._sort_indicator('stats', 3)}", justify="right", style="dim")

        # Apply filter if active
        if self._filter_text:
            stats_list = [s for s in stats_list if self._filter_text.lower() in s.rule.lower()]

        # Sort based on current settings or default to count descending
        if is_sorting:
            sort_keys = {
                0: lambda s: s.rule.lower(),
                1: lambda s: s.count,
                2: lambda s: s.mean_duration,
                3: lambda s: s.std_dev,
            }
            key_fn = sort_keys.get(self._sort_column, sort_keys[1])
            stats_list.sort(key=key_fn, reverse=not self._sort_ascending)
        else:
            stats_list.sort(key=lambda s: s.count, reverse=True)

        # Build hierarchical display: rule is primary, threads is secondary
        # Each row is (rule_display, threads_display, stats)
        # Limit total rows to 8 (not rules) to handle thread expansion
        max_rows = 8
        rows: list[tuple[str, str, RuleTimingStats]] = []
        for stats in stats_list:
            if len(rows) >= max_rows:
                break
            rule = stats.rule
            if rule in self._thread_stats and self._thread_stats[rule].stats_by_threads:
                # Has thread-specific data - show each thread count
                thread_stats = self._thread_stats[rule]
                sorted_threads = sorted(thread_stats.stats_by_threads.keys())
                for i, threads in enumerate(sorted_threads):
                    if len(rows) >= max_rows:
                        break
                    ts = thread_stats.stats_by_threads[threads]
                    # First row shows rule name, subsequent rows show blank
                    rule_display = rule if i == 0 else ""
                    rows.append((rule_display, str(threads), ts))
            else:
                # No thread data - show with "-" for threads
                rows.append((rule, "-", stats))

        for rule_display, threads_display, stats in rows:
            table.add_row(
                rule_display,
                threads_display,
                str(stats.count),
                format_duration(stats.mean_duration),
                format_duration(stats.std_dev) if stats.std_dev > 0 else "-",
            )

        title = "Rule Statistics (Historical)"
        if is_sorting:
            title += " [bold cyan]◀ sorting[/bold cyan]"
        border = f"bold {FG_BLUE}" if is_sorting else FG_BLUE
        return Panel(table, title=title, border_style=border, padding=0)

    def _make_footer(self) -> Panel:
        """Create the footer with settings and key bindings."""
        footer = Text()
        now = datetime.now().strftime("%H:%M:%S")

        footer.append(f"Updated: {now}", style="dim")
        footer.append("  │  ", style="dim")

        # Show log position (e.g., "Log: 1/10" or "Log: 3/10 [historical]")
        total_logs = len(self._available_logs)
        if total_logs > 0:
            current_pos = self._current_log_index + 1  # 1-indexed for display
            footer.append(f"Log: {current_pos}/{total_logs}", style=FG_BLUE)
            if self._current_log_index > 0:
                footer.append(" [historical]", style="yellow")
            footer.append("  │  ", style="dim")

        footer.append(f"Refresh: {self.refresh_rate}s", style="dim")
        footer.append("  │  ", style="dim")
        footer.append("ETA: ", style="dim")
        eta_style = FG_GREEN if self.use_estimation else "red"
        footer.append("ON" if self.use_estimation else "OFF", style=eta_style)
        footer.append("  │  ", style="dim")
        footer.append("Wildcard: ", style="dim")
        wc_style = FG_GREEN if self._use_wildcard_conditioning else "red"
        footer.append("ON" if self._use_wildcard_conditioning else "OFF", style=wc_style)
        footer.append("  │  ", style="dim")
        footer.append(f"Layout: {self._layout_mode.value}", style="dim")

        if self._filter_mode:
            footer.append("  │  ", style="dim")
            footer.append(f"Filter: /{self._filter_input}_", style="bold yellow")
        elif self._filter_text:
            footer.append("  │  ", style="dim")
            footer.append(f"Filter: {self._filter_text}", style="yellow")
            if self._filter_matches:
                match_info = f" ({self._filter_index + 1}/{len(self._filter_matches)})"
                footer.append(match_info, style="dim")

        if self._job_selection_mode:
            footer.append("  │  ", style="dim")
            footer.append("Job Log", style="bold cyan")
            footer.append(" (Esc exit)", style="dim")

        footer.append("  │  ")
        footer.append("snakesee", style=f"bold {FG_BLUE}")
        footer.append(" by ", style="dim")
        footer.append("Fulcrum Genomics", style=FG_BLUE)
        footer.append("  │  ")
        footer.append("?", style="bold")
        footer.append("=help", style="dim")

        return Panel(footer, border_style=FG_BLUE, padding=(0, 1))

    def _make_layout(
        self,
        progress: WorkflowProgress,
        estimate: TimeEstimate | None,
    ) -> Layout:
        """Create the complete TUI layout."""
        layout = Layout()

        # Easter egg takes over the whole screen
        if self._show_easter_egg:
            layout.split_column(
                Layout(name="easter_egg"),
                Layout(name="footer", size=3),
            )
            layout["easter_egg"].update(self._make_easter_egg_panel())
            layout["footer"].update(self._make_footer())
            return layout

        if self._show_help:
            # Show help overlay
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="help"),
                Layout(name="footer", size=3),
            )
            layout["header"].update(self._make_header(progress))
            layout["help"].update(self._make_help_panel())
            layout["footer"].update(self._make_footer())
            return layout

        if self._layout_mode == LayoutMode.MINIMAL:
            # Minimal: just header, progress, footer
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="progress", size=6),
                Layout(name="footer", size=3),
            )
            layout["header"].update(self._make_header(progress))
            layout["progress"].update(self._make_progress_panel(progress, estimate))
            layout["footer"].update(self._make_footer())

        elif self._layout_mode == LayoutMode.COMPACT:
            # Compact: header, progress, running jobs only
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="progress", size=6),
                Layout(name="running"),
                Layout(name="footer", size=3),
            )
            layout["header"].update(self._make_header(progress))
            layout["progress"].update(self._make_progress_panel(progress, estimate))
            layout["running"].update(self._make_running_table(progress))
            layout["footer"].update(self._make_footer())

        else:  # FULL
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="progress", size=6),
                Layout(name="body"),
                Layout(name="summary_footer", size=3),
                Layout(name="footer", size=3),
            )

            layout["body"].split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )

            # Check if we have incomplete jobs to show
            has_failed = bool(progress.failed_jobs_list)
            has_incomplete = bool(progress.incomplete_jobs_list)

            # Left column: running jobs, incomplete jobs (if any), pending jobs
            if has_incomplete:
                layout["left"].split_column(
                    Layout(name="running", ratio=1, minimum_size=3),
                    Layout(name="incomplete", ratio=1, minimum_size=3),
                    Layout(name="pending", ratio=1, minimum_size=3),
                )
            else:
                layout["left"].split_column(
                    Layout(name="running", ratio=1, minimum_size=3),
                    Layout(name="pending", ratio=1, minimum_size=3),
                )

            # Right column: completions, failed jobs (if any), stats
            if has_failed:
                layout["right"].split_column(
                    Layout(name="completions", ratio=1, minimum_size=3),
                    Layout(name="failed", ratio=1, minimum_size=3),
                    Layout(name="stats", ratio=1, minimum_size=3),
                )
                layout["failed"].update(self._make_failed_jobs_panel(progress))
            else:
                layout["right"].split_column(
                    Layout(name="completions", ratio=1, minimum_size=3),
                    Layout(name="stats", ratio=1, minimum_size=3),
                )

            layout["header"].update(self._make_header(progress))
            layout["progress"].update(self._make_progress_panel(progress, estimate))

            # Always show running jobs panel
            layout["running"].update(self._make_running_table(progress))

            # Show incomplete jobs panel if there are incomplete jobs
            if has_incomplete:
                layout["incomplete"].update(self._make_incomplete_jobs_panel(progress))

            # Show log panel when in job selection mode, otherwise show pending jobs
            if self._job_selection_mode:
                layout["pending"].update(self._make_job_log_panel(progress))
            else:
                layout["pending"].update(self._make_pending_jobs_panel(progress))
            layout["completions"].update(self._make_completions_table(progress))
            layout["stats"].update(self._make_stats_panel())
            layout["summary_footer"].update(self._make_summary_footer(progress))
            layout["footer"].update(self._make_footer())

        return layout

    def _get_cutoff_time(self) -> float | None:
        """Get the cutoff time for filtering (when the next log started)."""
        if self._current_log_index == 0:
            return None  # Latest log, no cutoff
        if self._current_log_index > 0 and len(self._available_logs) > 1:
            # Get the creation time of the next newer log
            next_log_index = self._current_log_index - 1
            if next_log_index >= 0:
                try:
                    return self._available_logs[next_log_index].stat().st_ctime
                except OSError:
                    pass
        return None

    def _poll_state(self) -> tuple[WorkflowProgress, TimeEstimate | None]:
        """Poll the current workflow state and estimate."""
        # Refresh log list if viewing latest
        if self._current_log_index == 0:
            self._refresh_log_list()

        # Get the selected log file and cutoff time for historical view
        log_file = self._get_current_log() if self._current_log_index > 0 else None
        self._cutoff_time = self._get_cutoff_time()

        # Read new events from logger plugin (if available)
        events = self._read_new_events()

        # Use incremental reader only for latest log (index 0)
        reader = self._log_reader if self._current_log_index == 0 else None

        progress = parse_workflow_state(
            self.workflow_dir,
            log_file=log_file,
            cutoff_time=self._cutoff_time,
            log_reader=reader,
        )

        # Validate: compare event-based state with parsed state (before applying)
        # This logs discrepancies to help find bugs in either approach
        if events:
            self._validate_state(events, progress)

        # Apply events to enhance progress accuracy
        if events:
            progress = self._apply_events_to_progress(progress, events)
            self._force_refresh = True

        # Update rule_stats with newly completed jobs (for Rule Statistics panel)
        self._update_rule_stats_from_completions(progress)

        estimate = None
        if self._estimator is not None:
            estimate = self._estimator.estimate_remaining(progress)

        return progress, estimate

    def run(self) -> None:
        """
        Run the TUI main loop.

        Continuously refreshes the display until the user presses 'q'
        or the workflow completes.
        """
        # Check if we're in a terminal that supports the TUI
        if not self.console.is_terminal:
            self.console.print(
                "[yellow]Warning:[/yellow] Not running in an interactive terminal. "
                "Use 'snakesee status' for non-interactive output.",
            )
            return

        try:
            import select
            import termios
            import tty
        except ImportError:
            # termios/tty not available (Windows without WSL)
            self._run_simple()
            return

        # Save terminal settings and switch to raw mode for single-key input
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            # Set terminal to raw mode (cbreak would also work)
            tty.setcbreak(fd)

            # Initial state
            progress, estimate = self._poll_state()

            with Live(
                self._make_layout(progress, estimate),
                console=self.console,
                refresh_per_second=1,
                screen=True,
            ) as live:
                last_update = time.time()

                while self._running:
                    # Check for keyboard input (non-blocking)
                    if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)

                        # Handle escape sequences (arrow keys, etc.)
                        if key == "\x1b":
                            # Check if this is an escape sequence or just Esc
                            if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                                seq = sys.stdin.read(2)
                                if seq == "[A":  # Up arrow
                                    key = "\x10"  # Map to Ctrl+p
                                elif seq == "[B":  # Down arrow
                                    key = "\x0e"  # Map to Ctrl+n
                                # Otherwise keep as Escape

                        if self._handle_key(key):
                            break

                    # Refresh at the specified rate or if forced (unless paused)
                    now = time.time()
                    should_refresh = self._force_refresh or (
                        not self._paused and now - last_update >= self.refresh_rate
                    )

                    if should_refresh:
                        if not self._paused or self._force_refresh:
                            progress, estimate = self._poll_state()
                        live.update(self._make_layout(progress, estimate))
                        last_update = now
                        self._force_refresh = False

                        # Auto-exit when workflow completes (optional)
                        if progress.status in (
                            WorkflowStatus.COMPLETED,
                            WorkflowStatus.FAILED,
                            WorkflowStatus.INCOMPLETE,
                        ):
                            # Show final state for a moment before potentially exiting
                            time.sleep(1)

        except KeyboardInterrupt:
            pass  # Clean exit on Ctrl+C
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _run_simple(self) -> None:
        """
        Simple run loop for environments without select().

        Falls back to a non-interactive refresh loop.
        """
        self.console.print("[yellow]Running in simple mode (no keyboard input)[/yellow]")
        self.console.print("Press Ctrl+C to exit\n")

        try:
            while self._running:
                progress, estimate = self._poll_state()

                # Clear and redraw
                self.console.clear()
                self.console.print(self._make_layout(progress, estimate))

                time.sleep(self.refresh_rate)

                # Auto-exit when workflow completes
                if progress.status in (
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.FAILED,
                    WorkflowStatus.INCOMPLETE,
                ):
                    self.console.print("\n[bold]Workflow finished.[/bold]")
                    break

        except KeyboardInterrupt:
            pass
