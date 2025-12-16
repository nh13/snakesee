"""Tests for the TUI module."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

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
