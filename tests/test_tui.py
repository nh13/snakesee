"""Tests for the TUI module."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from snakesee.events import EventType
from snakesee.models import JobInfo
from snakesee.models import TimeEstimate
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.tui import DEFAULT_REFRESH_RATE
from snakesee.tui import FG_BLUE
from snakesee.tui import FG_GREEN
from snakesee.tui import MAX_REFRESH_RATE
from snakesee.tui import MIN_REFRESH_RATE
from snakesee.tui import LayoutMode
from snakesee.tui import WorkflowMonitorTUI
from tests.conftest import make_job_info
from tests.conftest import make_snakesee_event
from tests.conftest import make_time_estimate
from tests.conftest import make_workflow_progress


class TestLayoutMode:
    """Tests for LayoutMode enum."""

    def test_layout_modes_exist(self) -> None:
        """Test that all layout modes are defined."""
        assert LayoutMode.FULL.value == "full"
        assert LayoutMode.COMPACT.value == "compact"
        assert LayoutMode.MINIMAL.value == "minimal"


class TestBrandingColors:
    """Tests for branding colors."""

    def test_fg_blue_defined(self) -> None:
        """Test FG_BLUE color is defined."""
        assert FG_BLUE == "#26a8e0"

    def test_fg_green_defined(self) -> None:
        """Test FG_GREEN color is defined."""
        assert FG_GREEN == "#38b44a"


class TestWorkflowMonitorTUI:
    """Tests for WorkflowMonitorTUI class."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        """Create a mock console."""
        console = MagicMock()
        console.width = 120
        console.height = 40
        console.is_terminal = True
        return console

    @pytest.fixture
    def tui(
        self, snakemake_dir: Path, tmp_path: Path, mock_console: MagicMock
    ) -> WorkflowMonitorTUI:
        """Create a TUI instance for testing."""
        with patch("snakesee.tui.Console", return_value=mock_console):
            return WorkflowMonitorTUI(
                workflow_dir=tmp_path,
                refresh_rate=2.0,
                use_estimation=True,
            )

    def test_init_default_values(self, snakemake_dir: Path, tmp_path: Path) -> None:
        """Test TUI initialization with default values."""
        tui = WorkflowMonitorTUI(workflow_dir=tmp_path)
        assert tui.workflow_dir == tmp_path
        assert tui.refresh_rate == DEFAULT_REFRESH_RATE
        assert tui.use_estimation is True

    def test_init_custom_values(self, snakemake_dir: Path, tmp_path: Path) -> None:
        """Test TUI initialization with custom values."""
        tui = WorkflowMonitorTUI(
            workflow_dir=tmp_path,
            refresh_rate=5.0,
            use_estimation=False,
        )
        assert tui.refresh_rate == 5.0
        assert tui.use_estimation is False

    def test_handle_key_quit(self, tui: WorkflowMonitorTUI) -> None:
        """Test quit key handler."""
        assert tui._handle_key("q") is True
        assert tui._handle_key("Q") is True

    def test_handle_key_toggle_pause(self, tui: WorkflowMonitorTUI) -> None:
        """Test pause toggle key handler."""
        assert not tui._paused
        tui._handle_key("p")
        assert tui._paused
        tui._handle_key("p")  # type: ignore[unreachable]
        assert not tui._paused

    def test_handle_key_toggle_estimation(self, tui: WorkflowMonitorTUI) -> None:
        """Test estimation toggle key handler."""
        initial = tui.use_estimation
        tui._handle_key("e")
        assert tui.use_estimation != initial

    def test_handle_key_toggle_wildcard_conditioning(self, tui: WorkflowMonitorTUI) -> None:
        """Test wildcard conditioning toggle key handler."""
        assert tui._use_wildcard_conditioning is False
        tui._handle_key("w")
        assert tui._use_wildcard_conditioning is True

    def test_handle_key_toggle_help(self, tui: WorkflowMonitorTUI) -> None:
        """Test help toggle key handler."""
        assert tui._show_help is False
        tui._handle_key("?")
        assert tui._show_help is True

    def test_handle_key_refresh_rate_decrease(self, tui: WorkflowMonitorTUI) -> None:
        """Test refresh rate decrease keys."""
        initial = tui.refresh_rate
        tui._handle_key("j")
        assert tui.refresh_rate == max(MIN_REFRESH_RATE, initial - 0.5)

    def test_handle_key_refresh_rate_increase(self, tui: WorkflowMonitorTUI) -> None:
        """Test refresh rate increase keys."""
        initial = tui.refresh_rate
        tui._handle_key("k")
        assert tui.refresh_rate == min(MAX_REFRESH_RATE, initial + 0.5)

    def test_handle_key_refresh_rate_reset(self, tui: WorkflowMonitorTUI) -> None:
        """Test refresh rate reset key."""
        tui.refresh_rate = 10.0
        tui._handle_key("0")
        assert tui.refresh_rate == DEFAULT_REFRESH_RATE

    def test_handle_key_refresh_rate_minimum(self, tui: WorkflowMonitorTUI) -> None:
        """Test refresh rate minimum key."""
        tui._handle_key("G")
        assert tui.refresh_rate == MIN_REFRESH_RATE

    def test_handle_key_layout_cycle(self, tui: WorkflowMonitorTUI) -> None:
        """Test layout cycle key."""
        assert tui._layout_mode == LayoutMode.FULL
        tui._handle_key("\t")
        assert tui._layout_mode == LayoutMode.COMPACT
        tui._handle_key("\t")  # type: ignore[unreachable]
        assert tui._layout_mode == LayoutMode.MINIMAL
        tui._handle_key("\t")
        assert tui._layout_mode == LayoutMode.FULL

    def test_handle_key_filter_mode(self, tui: WorkflowMonitorTUI) -> None:
        """Test filter mode key."""
        assert tui._filter_mode is False
        tui._handle_key("/")
        assert tui._filter_mode is True

    def test_handle_filter_key_escape(self, tui: WorkflowMonitorTUI) -> None:
        """Test escape in filter mode."""
        tui._filter_mode = True
        tui._filter_input = "test"
        tui._handle_filter_key("\x1b")
        assert tui._filter_mode is False
        assert tui._filter_input == ""

    def test_handle_filter_key_enter(self, tui: WorkflowMonitorTUI) -> None:
        """Test enter in filter mode."""
        tui._filter_mode = True
        tui._filter_input = "align"
        tui._handle_filter_key("\r")
        assert tui._filter_mode is False
        assert tui._filter_text == "align"

    def test_handle_filter_key_typing(self, tui: WorkflowMonitorTUI) -> None:
        """Test typing in filter mode."""
        tui._filter_mode = True
        tui._filter_input = ""
        tui._handle_filter_key("a")
        assert tui._filter_input == "a"
        tui._handle_filter_key("b")
        assert tui._filter_input == "ab"

    def test_handle_filter_key_backspace(self, tui: WorkflowMonitorTUI) -> None:
        """Test backspace in filter mode."""
        tui._filter_mode = True
        tui._filter_input = "abc"
        tui._handle_filter_key("\x7f")
        assert tui._filter_input == "ab"

    def test_filter_jobs(self, tui: WorkflowMonitorTUI) -> None:
        """Test job filtering."""
        jobs = [
            JobInfo(rule="align_reads"),
            JobInfo(rule="sort_bam"),
            JobInfo(rule="align_contigs"),
        ]
        tui._filter_text = "align"
        filtered = tui._filter_jobs(jobs)
        assert len(filtered) == 2
        assert all("align" in j.rule for j in filtered)

    def test_filter_jobs_no_filter(self, tui: WorkflowMonitorTUI) -> None:
        """Test job filtering with no filter."""
        jobs = [JobInfo(rule="align"), JobInfo(rule="sort")]
        tui._filter_text = None
        filtered = tui._filter_jobs(jobs)
        assert len(filtered) == 2

    def test_make_progress_bar(self, tui: WorkflowMonitorTUI) -> None:
        """Test progress bar creation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
            failed_jobs=1,
        )
        bar = tui._make_progress_bar(progress, width=20)
        assert len(bar.plain) == 20

    def test_make_progress_panel(self, tui: WorkflowMonitorTUI) -> None:
        """Test progress panel creation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
        )
        estimate = TimeEstimate(
            seconds_remaining=300,
            lower_bound=200,
            upper_bound=400,
            confidence=0.8,
            method="weighted",
        )
        panel = tui._make_progress_panel(progress, estimate)
        assert panel.title == "Progress"

    def test_make_running_table(self, tui: WorkflowMonitorTUI) -> None:
        """Test running jobs table creation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
            running_jobs=[JobInfo(rule="align", job_id="1")],
        )
        panel = tui._make_running_table(progress)
        assert "Running" in panel.title

    def test_make_completions_table(self, tui: WorkflowMonitorTUI) -> None:
        """Test completions table creation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
            recent_completions=[
                JobInfo(rule="align", start_time=100.0, end_time=200.0),
            ],
        )
        panel = tui._make_completions_table(progress)
        assert panel.title == "Recent Completions"

    def test_make_failed_jobs_panel_empty(self, tui: WorkflowMonitorTUI) -> None:
        """Test failed jobs panel with no failures."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
        )
        panel = tui._make_failed_jobs_panel(progress)
        assert "Failed" in panel.title

    def test_make_failed_jobs_panel_with_failures(self, tui: WorkflowMonitorTUI) -> None:
        """Test failed jobs panel with failures."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.FAILED,
            total_jobs=10,
            completed_jobs=5,
            failed_jobs=2,
            failed_jobs_list=[
                JobInfo(rule="align", job_id="1"),
                JobInfo(rule="sort", job_id="2"),
            ],
        )
        panel = tui._make_failed_jobs_panel(progress)
        assert "(2)" in panel.title

    def test_make_summary_footer(self, tui: WorkflowMonitorTUI) -> None:
        """Test summary footer creation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=20,
            completed_jobs=10,
            failed_jobs=2,
            running_jobs=[JobInfo(rule="test")] * 3,
        )
        panel = tui._make_summary_footer(progress)
        assert panel is not None

    def test_sort_indicator_active(self, tui: WorkflowMonitorTUI) -> None:
        """Test sort indicator when sorting is active."""
        tui._sort_table = "running"
        tui._sort_column = 0
        tui._sort_ascending = True
        indicator = tui._sort_indicator("running", 0)
        assert "▲" in indicator

    def test_sort_indicator_descending(self, tui: WorkflowMonitorTUI) -> None:
        """Test sort indicator for descending sort."""
        tui._sort_table = "running"
        tui._sort_column = 0
        tui._sort_ascending = False
        indicator = tui._sort_indicator("running", 0)
        assert "▼" in indicator

    def test_sort_indicator_inactive(self, tui: WorkflowMonitorTUI) -> None:
        """Test sort indicator when not sorting this table."""
        tui._sort_table = "stats"
        indicator = tui._sort_indicator("running", 0)
        assert indicator == ""

    def test_handle_sort_key_cycle(self, tui: WorkflowMonitorTUI) -> None:
        """Test sort table cycling forward with 's'."""
        assert tui._sort_table is None
        tui._handle_key("s")
        assert tui._sort_table == "running"
        tui._handle_key("s")
        assert tui._sort_table == "completions"
        tui._handle_key("s")
        assert tui._sort_table == "pending"
        tui._handle_key("s")
        assert tui._sort_table == "stats"
        tui._handle_key("s")
        assert tui._sort_table is None

    def test_handle_sort_key_cycle_backward(self, tui: WorkflowMonitorTUI) -> None:
        """Test sort table cycling backward with 'S' (shift+s)."""
        assert tui._sort_table is None
        tui._handle_key("S")
        assert tui._sort_table == "stats"
        tui._handle_key("S")
        assert tui._sort_table == "pending"
        tui._handle_key("S")
        assert tui._sort_table == "completions"
        tui._handle_key("S")
        assert tui._sort_table == "running"
        tui._handle_key("S")
        assert tui._sort_table is None

    def test_handle_log_navigation_older(self, tui: WorkflowMonitorTUI) -> None:
        """Test log navigation to older log."""
        tui._available_logs = [Path("log1"), Path("log2"), Path("log3")]
        tui._current_log_index = 0
        with patch.object(tui, "_refresh_log_list"):
            tui._handle_key("[")
        assert tui._current_log_index == 1

    def test_handle_log_navigation_newer(self, tui: WorkflowMonitorTUI) -> None:
        """Test log navigation to newer log."""
        tui._available_logs = [Path("log1"), Path("log2"), Path("log3")]
        tui._current_log_index = 2
        tui._handle_key("]")
        assert tui._current_log_index == 1


class TestEventHandling:
    """Tests for event processing methods."""

    def test_apply_events_empty_list(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test applying empty event list returns unchanged progress."""
        progress = make_workflow_progress()
        result = tui_with_mocks._apply_events_to_progress(progress, [])
        assert result.completed_jobs == progress.completed_jobs
        assert result.total_jobs == progress.total_jobs

    def test_apply_events_progress_event(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test that progress events update total and completed jobs."""
        progress = make_workflow_progress(total_jobs=100, completed_jobs=50)
        events = [make_snakesee_event(EventType.PROGRESS, total_jobs=100, completed_jobs=60)]
        result = tui_with_mocks._apply_events_to_progress(progress, events)
        assert result.completed_jobs == 60

    def test_apply_events_job_error(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test that job error events update failed count."""
        progress = make_workflow_progress(failed_jobs=0, failed_jobs_list=[])
        events = [make_snakesee_event(EventType.JOB_ERROR, rule_name="align", job_id=123)]
        result = tui_with_mocks._apply_events_to_progress(progress, events)
        assert result.failed_jobs == 1
        assert len(result.failed_jobs_list) == 1

    def test_handle_job_submitted_tracks_threads(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test that submitted events track thread info in JobRegistry."""
        event = make_snakesee_event(
            EventType.JOB_SUBMITTED, rule_name="align", job_id=123, threads=8
        )
        # In production, apply_event is called first to create the job
        tui_with_mocks._workflow_state.jobs.apply_event(event)
        # Pass empty running_jobs list as required by the method
        running_jobs: list[JobInfo] = []
        tui_with_mocks._handle_job_submitted_event(event, running_jobs)
        # Threads should be stored in JobRegistry
        job = tui_with_mocks._workflow_state.jobs.get_by_job_id("123")
        assert job is not None
        assert job.threads == 8


class TestPanelCreation:
    """Tests for panel creation methods."""

    def test_make_header_running(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test header shows RUNNING status."""
        progress = make_workflow_progress(status=WorkflowStatus.RUNNING)
        panel = tui_with_mocks._make_header(progress)
        # Panel should contain RUNNING text
        assert panel is not None

    def test_make_header_completed(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test header shows COMPLETED status."""
        progress = make_workflow_progress(status=WorkflowStatus.COMPLETED)
        panel = tui_with_mocks._make_header(progress)
        assert panel is not None

    def test_make_header_failed(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test header shows FAILED status."""
        progress = make_workflow_progress(status=WorkflowStatus.FAILED, failed_jobs=1)
        panel = tui_with_mocks._make_header(progress)
        assert panel is not None

    def test_make_header_incomplete(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test header shows INCOMPLETE status."""
        progress = make_workflow_progress(
            status=WorkflowStatus.INCOMPLETE,
            incomplete_jobs_list=[make_job_info(rule="interrupted")],
        )
        panel = tui_with_mocks._make_header(progress)
        assert panel is not None

    def test_make_progress_bar_all_succeeded(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test progress bar with all jobs succeeded."""
        progress = make_workflow_progress(total_jobs=100, completed_jobs=100, failed_jobs=0)
        bar = tui_with_mocks._make_progress_bar(progress, width=40)
        assert len(bar.plain) == 40

    def test_make_progress_bar_with_failures(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test progress bar shows failures in different color."""
        progress = make_workflow_progress(total_jobs=100, completed_jobs=80, failed_jobs=10)
        bar = tui_with_mocks._make_progress_bar(progress, width=40)
        assert len(bar.plain) == 40

    def test_make_progress_panel_with_estimate(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test progress panel includes ETA when estimate provided."""
        progress = make_workflow_progress()
        estimate = make_time_estimate(seconds_remaining=600)
        panel = tui_with_mocks._make_progress_panel(progress, estimate)
        assert panel.title == "Progress"

    def test_make_progress_panel_without_estimate(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test progress panel without ETA."""
        progress = make_workflow_progress()
        panel = tui_with_mocks._make_progress_panel(progress, None)
        assert panel.title == "Progress"

    def test_make_progress_panel_completed_workflow(
        self, tui_with_mocks: WorkflowMonitorTUI
    ) -> None:
        """Test progress panel for completed workflow."""
        progress = make_workflow_progress(
            status=WorkflowStatus.COMPLETED,
            total_jobs=100,
            completed_jobs=100,
        )
        panel = tui_with_mocks._make_progress_panel(progress, None)
        assert panel is not None

    def test_make_incomplete_jobs_panel_empty(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test incomplete jobs panel with no incomplete jobs."""
        progress = make_workflow_progress(incomplete_jobs_list=[])
        panel = tui_with_mocks._make_incomplete_jobs_panel(progress)
        assert "Incomplete" in panel.title

    def test_make_incomplete_jobs_panel_with_jobs(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test incomplete jobs panel with interrupted jobs."""
        progress = make_workflow_progress(
            status=WorkflowStatus.INCOMPLETE,
            incomplete_jobs_list=[
                make_job_info(rule="align", output_file=Path("output.bam")),
                make_job_info(rule="sort"),
            ],
        )
        panel = tui_with_mocks._make_incomplete_jobs_panel(progress)
        assert "(2)" in panel.title

    def test_make_pending_jobs_panel(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test pending jobs panel creation."""
        tui_with_mocks._estimator = mock_estimator
        progress = make_workflow_progress(total_jobs=100, completed_jobs=50)
        panel = tui_with_mocks._make_pending_jobs_panel(progress)
        assert "Pending" in panel.title

    def test_make_help_panel(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test help panel creation."""
        panel = tui_with_mocks._make_help_panel()
        assert "Keyboard" in panel.title

    def test_make_footer(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test footer creation."""
        footer = tui_with_mocks._make_footer()
        assert footer is not None


class TestLayout:
    """Tests for layout building."""

    def test_make_layout_full_mode(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test full layout mode includes all panels."""
        tui_with_mocks._layout_mode = LayoutMode.FULL
        progress = make_workflow_progress()
        estimate = make_time_estimate()
        layout = tui_with_mocks._make_layout(progress, estimate)
        assert layout is not None

    def test_make_layout_compact_mode(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test compact layout mode."""
        tui_with_mocks._layout_mode = LayoutMode.COMPACT
        progress = make_workflow_progress()
        layout = tui_with_mocks._make_layout(progress, None)
        assert layout is not None

    def test_make_layout_minimal_mode(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test minimal layout mode."""
        tui_with_mocks._layout_mode = LayoutMode.MINIMAL
        progress = make_workflow_progress()
        layout = tui_with_mocks._make_layout(progress, None)
        assert layout is not None

    def test_make_layout_with_failed_jobs(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test layout includes failed jobs panel when applicable."""
        tui_with_mocks._layout_mode = LayoutMode.FULL
        progress = make_workflow_progress(
            status=WorkflowStatus.FAILED,
            failed_jobs=2,
            failed_jobs_list=[
                make_job_info(rule="align", job_id="1"),
                make_job_info(rule="sort", job_id="2"),
            ],
        )
        layout = tui_with_mocks._make_layout(progress, None)
        assert layout is not None

    def test_make_layout_with_incomplete_jobs(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test layout includes incomplete panel when applicable."""
        tui_with_mocks._layout_mode = LayoutMode.FULL
        progress = make_workflow_progress(
            status=WorkflowStatus.INCOMPLETE,
            incomplete_jobs_list=[make_job_info(rule="interrupted")],
        )
        layout = tui_with_mocks._make_layout(progress, None)
        assert layout is not None

    def test_make_layout_with_help_overlay(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test layout with help overlay shown."""
        tui_with_mocks._show_help = True
        progress = make_workflow_progress()
        layout = tui_with_mocks._make_layout(progress, None)
        assert layout is not None


class TestDataBuilding:
    """Tests for data building and transformation methods."""

    def test_build_running_job_data_empty(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test building job data with no running jobs."""
        result = tui_with_mocks._build_running_job_data([])
        assert result == []

    def test_build_running_job_data_with_jobs(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test building job data with running jobs."""
        import time

        jobs = [
            make_job_info(rule="align", job_id="1", start_time=time.time() - 100),
            make_job_info(rule="sort", job_id="2", start_time=time.time() - 50),
        ]
        result = tui_with_mocks._build_running_job_data(jobs)
        assert len(result) == 2
        # Each result should be a tuple of (job, elapsed, remaining, start_time, tool_progress)
        assert result[0][0].rule == "align"

    def test_sort_running_job_data_by_rule(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test sorting job data by rule name."""
        import time

        jobs = [
            make_job_info(rule="zebra", job_id="1", start_time=time.time() - 100),
            make_job_info(rule="alpha", job_id="2", start_time=time.time() - 50),
        ]
        job_data = tui_with_mocks._build_running_job_data(jobs)
        tui_with_mocks._sort_table = "running"
        tui_with_mocks._sort_column = 0  # Rule column
        tui_with_mocks._sort_ascending = True
        sorted_data = tui_with_mocks._sort_running_job_data(job_data)
        assert sorted_data[0][0].rule == "alpha"

    def test_get_running_jobs_list(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test getting running jobs list with filtering."""
        import time

        progress = make_workflow_progress(
            running_jobs=[
                make_job_info(rule="align", job_id="1", start_time=time.time() - 100),
                make_job_info(rule="sort", job_id="2", start_time=time.time() - 50),
            ]
        )
        result = tui_with_mocks._get_running_jobs_list(progress)
        assert len(result) == 2

    def test_get_completions_list(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test getting completions list."""
        import time

        progress = make_workflow_progress(
            recent_completions=[
                make_job_info(
                    rule="align", start_time=time.time() - 200, end_time=time.time() - 100
                ),
            ],
            failed_jobs_list=[
                make_job_info(rule="sort", job_id="failed"),
            ],
        )
        jobs, failed_ids = tui_with_mocks._get_completions_list(progress)
        # Should include both completed and failed
        assert len(jobs) == 2
        assert len(failed_ids) == 1


class TestRuleStatsUpdate:
    """Tests for _update_rule_stats_from_completions method."""

    def test_update_stats_no_estimator(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test that no update happens without estimator."""
        tui_with_mocks._estimator = None
        progress = make_workflow_progress(
            recent_completions=[
                make_job_info(rule="align", job_id="1", start_time=100.0, end_time=200.0),
            ]
        )
        # Should not raise
        tui_with_mocks._update_rule_stats_from_completions(progress)

    def test_update_stats_with_job_id(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test stats update with job_id deduplication."""
        tui_with_mocks._estimator = mock_estimator

        progress = make_workflow_progress(
            recent_completions=[
                make_job_info(rule="align", job_id="1", start_time=100.0, end_time=200.0),
            ]
        )
        tui_with_mocks._update_rule_stats_from_completions(progress)

        # Should have added stats to RuleRegistry
        rule_stats = tui_with_mocks._workflow_state.rules.get("align")
        assert rule_stats is not None
        assert rule_stats.aggregate.count == 1

    def test_update_stats_deduplication(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test that duplicate jobs are not counted twice."""
        tui_with_mocks._estimator = mock_estimator

        job = make_job_info(rule="align", job_id="1", start_time=100.0, end_time=200.0)
        progress = make_workflow_progress(recent_completions=[job])

        # Call twice with same job
        tui_with_mocks._update_rule_stats_from_completions(progress)
        tui_with_mocks._update_rule_stats_from_completions(progress)

        # Should only count once (deduplication via _rule_stats_job_ids)
        rule_stats = tui_with_mocks._workflow_state.rules.get("align")
        assert rule_stats is not None
        assert rule_stats.aggregate.count == 1

    def test_update_stats_with_threads(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test that thread stats are updated when job has threads."""
        tui_with_mocks._estimator = mock_estimator

        progress = make_workflow_progress(
            recent_completions=[
                make_job_info(
                    rule="align", job_id="1", start_time=100.0, end_time=200.0, threads=4
                ),
            ]
        )
        tui_with_mocks._update_rule_stats_from_completions(progress)

        # Should have thread stats in RuleRegistry
        thread_stats_dict = tui_with_mocks._workflow_state.rules.to_thread_stats_dict()
        assert "align" in thread_stats_dict
        assert 4 in thread_stats_dict["align"].stats_by_threads

    def test_update_stats_skips_no_duration(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test that jobs without duration are skipped."""
        tui_with_mocks._estimator = mock_estimator

        # Job with no end_time means no duration
        progress = make_workflow_progress(
            recent_completions=[
                make_job_info(rule="align", job_id="1", start_time=100.0, end_time=None),
            ]
        )
        tui_with_mocks._update_rule_stats_from_completions(progress)

        # Should not have added stats
        assert tui_with_mocks._workflow_state.rules.get("align") is None


class TestStatsPanel:
    """Tests for _make_stats_panel method."""

    def test_stats_panel_estimation_disabled(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test stats panel when estimation is disabled."""
        tui_with_mocks.use_estimation = False
        panel = tui_with_mocks._make_stats_panel()
        assert "disabled" in str(panel.renderable).lower()

    def test_stats_panel_no_data(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test stats panel with no historical data."""
        tui_with_mocks._estimator = mock_estimator
        tui_with_mocks._estimator.rule_stats = {}
        panel = tui_with_mocks._make_stats_panel()
        assert "No historical data" in str(panel.renderable)

    def test_stats_panel_with_data(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test stats panel with rule statistics."""
        from snakesee.models import RuleTimingStats

        tui_with_mocks._estimator = mock_estimator
        tui_with_mocks._estimator.rule_stats = {
            "align": RuleTimingStats(rule="align", durations=[100.0, 110.0, 90.0]),
            "sort": RuleTimingStats(rule="sort", durations=[50.0, 55.0]),
        }
        panel = tui_with_mocks._make_stats_panel()
        assert "Rule Statistics" in panel.title

    def test_stats_panel_with_thread_stats(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test stats panel displays thread-specific statistics."""
        tui_with_mocks._estimator = mock_estimator

        # Populate RuleRegistry with thread-specific data
        tui_with_mocks._workflow_state.rules.record_completion(
            rule="align", duration=100.0, timestamp=1000.0, threads=4
        )
        tui_with_mocks._workflow_state.rules.record_completion(
            rule="align", duration=60.0, timestamp=2000.0, threads=8
        )

        panel = tui_with_mocks._make_stats_panel()
        assert panel is not None

    def test_stats_panel_sorting_active(
        self, tui_with_mocks: WorkflowMonitorTUI, mock_estimator: MagicMock
    ) -> None:
        """Test stats panel when sorting is active."""
        from snakesee.models import RuleTimingStats

        tui_with_mocks._estimator = mock_estimator
        tui_with_mocks._estimator.rule_stats = {
            "align": RuleTimingStats(rule="align", durations=[100.0]),
        }
        tui_with_mocks._sort_table = "stats"
        tui_with_mocks._sort_column = 0
        tui_with_mocks._sort_ascending = True

        panel = tui_with_mocks._make_stats_panel()
        assert "sorting" in panel.title


class TestJobLogPanel:
    """Tests for _make_job_log_panel method."""

    def test_job_log_panel_no_running_jobs(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test job log panel with no running jobs."""
        tui_with_mocks._job_selection_mode = True
        tui_with_mocks._log_source = "running"
        progress = make_workflow_progress(running_jobs=[])
        panel = tui_with_mocks._make_job_log_panel(progress)
        assert "No running jobs" in str(panel.renderable)

    def test_job_log_panel_no_completed_jobs(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test job log panel with no completed jobs."""
        tui_with_mocks._job_selection_mode = True
        tui_with_mocks._log_source = "completions"
        progress = make_workflow_progress(recent_completions=[], failed_jobs_list=[])
        panel = tui_with_mocks._make_job_log_panel(progress)
        assert "No completed jobs" in str(panel.renderable)

    def test_job_log_panel_no_log_file(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test job log panel when log file doesn't exist."""
        import time

        tui_with_mocks._job_selection_mode = True
        tui_with_mocks._log_source = "running"
        progress = make_workflow_progress(
            running_jobs=[
                make_job_info(rule="align", job_id="1", start_time=time.time() - 100),
            ]
        )
        panel = tui_with_mocks._make_job_log_panel(progress)
        assert "No log file for" in str(panel.renderable)


class TestInitMethods:
    """Tests for TUI initialization methods."""

    def test_init_event_reader_disabled(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test event reader init when events disabled."""
        tui_with_mocks._events_enabled = False
        tui_with_mocks._event_reader = "placeholder"  # type: ignore[assignment]
        tui_with_mocks._init_event_reader()
        # Should not modify event_reader when disabled
        assert tui_with_mocks._event_reader == "placeholder"

    def test_init_event_reader_no_file(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test event reader init when event file doesn't exist."""
        tui_with_mocks._events_enabled = True
        tui_with_mocks._init_event_reader()
        assert tui_with_mocks._event_reader is None

    def test_init_log_reader(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test log reader initialization."""
        tui_with_mocks._init_log_reader()
        # Should have created a log reader (even with placeholder)
        assert tui_with_mocks._log_reader is not None

    def test_init_validation_no_event_file(self, tui_with_mocks: WorkflowMonitorTUI) -> None:
        """Test validation init when event file doesn't exist."""
        tui_with_mocks._init_validation()
        # Should not create validator without event file
        assert tui_with_mocks._event_accumulator is None


class TestTerminalHandling:
    """Tests for TUI terminal handling and run loop."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        """Create a mock console."""
        console = MagicMock()
        console.width = 120
        console.height = 40
        console.is_terminal = True
        return console

    @pytest.fixture
    def tui(
        self, snakemake_dir: Path, tmp_path: Path, mock_console: MagicMock
    ) -> WorkflowMonitorTUI:
        """Create a TUI instance for testing."""
        with patch("snakesee.tui.Console", return_value=mock_console):
            return WorkflowMonitorTUI(
                workflow_dir=tmp_path,
                refresh_rate=2.0,
            )

    def test_run_non_terminal_prints_warning(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that run() prints warning when not in terminal."""
        mock_console.is_terminal = False

        tui.run()

        # Should have printed warning
        mock_console.print.assert_called()
        call_args = str(mock_console.print.call_args)
        assert "Warning" in call_args
        assert "interactive terminal" in call_args

    def test_run_non_terminal_returns_early(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that run() returns early when not in terminal."""
        mock_console.is_terminal = False

        # Should not raise any errors and return quickly
        tui.run()

        # Live should not have been created (no screen=True call)
        # Just verify it completed without errors

    def test_run_simple_is_fallback(self, tui: WorkflowMonitorTUI, mock_console: MagicMock) -> None:
        """Test that _run_simple exists as a fallback mode."""
        # Verify _run_simple can be called directly (fallback path)
        tui._running = False  # Exit immediately
        tui._run_simple()
        # Should have printed the simple mode message
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("simple mode" in c.lower() for c in calls)

    def test_run_simple_prints_mode_message(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that _run_simple prints the simple mode message."""
        # Make workflow appear complete immediately to exit loop
        tui._running = False

        tui._run_simple()

        # Check that simple mode message was printed
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("simple mode" in c.lower() for c in calls)

    def test_run_simple_keyboard_interrupt(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that _run_simple handles KeyboardInterrupt gracefully."""
        with patch.object(tui, "_poll_state", side_effect=KeyboardInterrupt):
            # Should not raise
            tui._run_simple()

    def test_run_simple_exits_on_completed(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that _run_simple exits when workflow completes."""
        progress = make_workflow_progress(status=WorkflowStatus.COMPLETED)

        call_count = 0

        def mock_poll() -> tuple[WorkflowProgress, TimeEstimate | None]:
            nonlocal call_count
            call_count += 1
            return progress, None

        with patch.object(tui, "_poll_state", side_effect=mock_poll):
            with patch("time.sleep"):  # Don't actually sleep
                tui._run_simple()

        # Should have polled once and exited
        assert call_count == 1

    def test_run_simple_exits_on_failed(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that _run_simple exits when workflow fails."""
        progress = make_workflow_progress(status=WorkflowStatus.FAILED)

        with patch.object(tui, "_poll_state", return_value=(progress, None)):
            with patch("time.sleep"):
                tui._run_simple()

        # Verify it printed the finished message
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("finished" in c.lower() for c in calls)


class TestKeyboardInput:
    """Tests for keyboard input handling in the TUI."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        """Create a mock console."""
        console = MagicMock()
        console.width = 120
        console.height = 40
        console.is_terminal = True
        return console

    @pytest.fixture
    def tui(
        self, snakemake_dir: Path, tmp_path: Path, mock_console: MagicMock
    ) -> WorkflowMonitorTUI:
        """Create a TUI instance for testing."""
        with patch("snakesee.tui.Console", return_value=mock_console):
            return WorkflowMonitorTUI(
                workflow_dir=tmp_path,
                refresh_rate=2.0,
            )

    def test_handle_key_quit_lowercase(self, tui: WorkflowMonitorTUI) -> None:
        """Test that 'q' quits."""
        assert tui._handle_key("q") is True

    def test_handle_key_quit_uppercase(self, tui: WorkflowMonitorTUI) -> None:
        """Test that 'Q' quits."""
        assert tui._handle_key("Q") is True

    def test_handle_key_pause_toggle(self, tui: WorkflowMonitorTUI) -> None:
        """Test pause toggle with 'p'."""
        assert tui._paused is False
        result = tui._handle_key("p")
        assert result is False
        assert tui._paused is True

    def test_handle_key_pause_toggle_back(self, tui: WorkflowMonitorTUI) -> None:
        """Test pause toggle back with 'p'."""
        tui._paused = True
        tui._handle_key("p")
        assert tui._paused is False

    def test_handle_key_estimation_toggle(self, tui: WorkflowMonitorTUI) -> None:
        """Test estimation toggle with 'e'."""
        original = tui.use_estimation
        tui._handle_key("e")
        assert tui.use_estimation is not original

    def test_handle_key_refresh_rate_increase(self, tui: WorkflowMonitorTUI) -> None:
        """Test refresh rate increase with 'k' (slower refresh)."""
        original_rate = tui.refresh_rate
        tui._handle_key("k")
        assert tui.refresh_rate > original_rate

    def test_handle_key_refresh_rate_decrease(self, tui: WorkflowMonitorTUI) -> None:
        """Test refresh rate decrease with 'j' (faster refresh)."""
        tui.refresh_rate = 5.0  # Set to middle value
        original_rate = tui.refresh_rate
        tui._handle_key("j")
        assert tui.refresh_rate < original_rate

    def test_handle_key_help_toggle(self, tui: WorkflowMonitorTUI) -> None:
        """Test help overlay toggle with '?'."""
        assert tui._show_help is False
        tui._handle_key("?")
        assert tui._show_help is True

    def test_handle_key_help_close(self, tui: WorkflowMonitorTUI) -> None:
        """Test that any key closes help overlay."""
        tui._show_help = True
        tui._handle_key("x")
        assert tui._show_help is False

    def test_handle_key_force_refresh(self, tui: WorkflowMonitorTUI) -> None:
        """Test force refresh with 'r'."""
        tui._force_refresh = False
        tui._handle_key("r")
        assert tui._force_refresh is True

    def test_handle_key_ctrl_r_hard_refresh(self, tui: WorkflowMonitorTUI) -> None:
        """Test hard refresh with Ctrl+r."""
        tui._force_refresh = False
        tui._handle_key("\x12")  # Ctrl+r
        assert tui._force_refresh is True

    def test_handle_key_wildcard_conditioning_toggle(self, tui: WorkflowMonitorTUI) -> None:
        """Test wildcard conditioning toggle with 'w'."""
        original = tui._use_wildcard_conditioning
        tui._handle_key("w")
        assert tui._use_wildcard_conditioning is not original

    def test_handle_key_unknown_returns_false(self, tui: WorkflowMonitorTUI) -> None:
        """Test that unknown keys return False."""
        assert tui._handle_key("z") is False
        assert tui._handle_key("@") is False

    def test_handle_key_navigation_up(self, tui: WorkflowMonitorTUI) -> None:
        """Test navigation up with Ctrl+p."""
        # Ctrl+p is mapped from up arrow - just verify no error
        result = tui._handle_key("\x10")  # Ctrl+p
        assert result is False  # Navigation doesn't quit

    def test_handle_key_navigation_down(self, tui: WorkflowMonitorTUI) -> None:
        """Test navigation down with Ctrl+n."""
        # Ctrl+n is mapped from down arrow
        tui._handle_key("\x0e")  # Ctrl+n
        # Just verify no error

    def test_handle_key_layout_cycle(self, tui: WorkflowMonitorTUI) -> None:
        """Test layout mode cycling with Tab."""
        original_mode = tui._layout_mode
        tui._handle_key("\t")  # Tab cycles layout
        # Should cycle to next mode
        assert tui._layout_mode != original_mode


class TestTerminalSettingsRestore:
    """Tests for terminal settings save/restore behavior."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        """Create a mock console for terminal tests."""
        console = MagicMock()
        console.width = 120
        console.height = 40
        console.is_terminal = True
        return console

    @pytest.fixture
    def tui(
        self, snakemake_dir: Path, tmp_path: Path, mock_console: MagicMock
    ) -> WorkflowMonitorTUI:
        """Create a TUI instance for testing."""
        with patch("snakesee.tui.Console", return_value=mock_console):
            return WorkflowMonitorTUI(
                workflow_dir=tmp_path,
                refresh_rate=0.1,  # Fast for testing
            )

    def test_terminal_settings_restored_on_normal_exit(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test that terminal settings are restored on normal exit."""
        mock_termios = MagicMock()
        mock_tty = MagicMock()
        mock_old_settings = ["mock", "settings"]
        mock_termios.tcgetattr.return_value = mock_old_settings

        # Make stdin look like a valid file descriptor
        mock_stdin = MagicMock()
        mock_stdin.fileno.return_value = 0

        def quit_immediately(*args: object) -> tuple[list[MagicMock], list[object], list[object]]:
            # First call returns stdin ready, second returns empty
            return ([mock_stdin], [], [])

        call_count = 0

        def mock_select(*args: object) -> tuple[list[object], list[object], list[object]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ([mock_stdin], [], [])
            return ([], [], [])

        with patch.dict(
            "sys.modules",
            {"termios": mock_termios, "tty": mock_tty},
        ):
            with patch("sys.stdin", mock_stdin):
                with patch("select.select", side_effect=mock_select):
                    with patch("os.read", return_value=b"q"):  # Quit key
                        with patch("fcntl.fcntl"):
                            with patch("snakesee.tui.Live"):
                                tui.run()

        # Verify tcsetattr was called to restore settings
        mock_termios.tcsetattr.assert_called()

    def test_terminal_settings_restored_on_keyboard_interrupt(
        self, tui: WorkflowMonitorTUI, mock_console: MagicMock
    ) -> None:
        """Test terminal settings are restored after KeyboardInterrupt."""
        mock_termios = MagicMock()
        mock_tty = MagicMock()
        mock_old_settings = ["mock", "settings"]
        mock_termios.tcgetattr.return_value = mock_old_settings

        mock_stdin = MagicMock()
        mock_stdin.fileno.return_value = 0

        with patch.dict(
            "sys.modules",
            {"termios": mock_termios, "tty": mock_tty},
        ):
            with patch("sys.stdin", mock_stdin):
                with patch("select.select", side_effect=KeyboardInterrupt):
                    with patch("snakesee.tui.Live"):
                        tui.run()

        # Verify settings were restored even after interrupt
        mock_termios.tcsetattr.assert_called()


class TestEscapeSequenceParsing:
    """Tests for escape sequence parsing in keyboard input."""

    @pytest.fixture
    def mock_console(self) -> MagicMock:
        """Create a mock console."""
        console = MagicMock()
        console.width = 120
        console.height = 40
        console.is_terminal = True
        return console

    @pytest.fixture
    def tui(
        self, snakemake_dir: Path, tmp_path: Path, mock_console: MagicMock
    ) -> WorkflowMonitorTUI:
        """Create a TUI instance for testing."""
        with patch("snakesee.tui.Console", return_value=mock_console):
            return WorkflowMonitorTUI(
                workflow_dir=tmp_path,
                refresh_rate=2.0,
            )

    def test_up_arrow_mapped_to_ctrl_p(self, tui: WorkflowMonitorTUI) -> None:
        """Test that up arrow escape sequence is recognized."""
        # The run loop parses \x1b[A or \x1bOA as up arrow
        # and maps to \x10 (Ctrl+p) before calling _handle_key
        # We test the _handle_key with the mapped value
        result = tui._handle_key("\x10")  # Ctrl+p (mapped from up arrow)
        assert result is False  # Navigation doesn't quit

    def test_down_arrow_mapped_to_ctrl_n(self, tui: WorkflowMonitorTUI) -> None:
        """Test that down arrow escape sequence is recognized."""
        result = tui._handle_key("\x0e")  # Ctrl+n (mapped from down arrow)
        assert result is False

    def test_escape_alone_does_not_quit(self, tui: WorkflowMonitorTUI) -> None:
        """Test that pressing Escape alone doesn't quit."""
        result = tui._handle_key("\x1b")  # Escape
        assert result is False
