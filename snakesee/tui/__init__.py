"""Terminal User Interface for Snakemake workflow monitoring.

This package provides a Rich-based TUI for monitoring Snakemake workflows:

- WorkflowMonitorTUI: Main TUI class with full-screen monitoring
- LayoutMode: Available layout modes (full/compact/minimal)

The TUI provides:
- Real-time progress tracking with time estimates
- Running, completed, and failed job display
- Log file viewing with scrolling
- Keyboard-driven filtering and navigation

Usage:
    from snakesee.tui import WorkflowMonitorTUI

    tui = WorkflowMonitorTUI(workflow_dir=Path("."))
    tui.run()

The module structure is:
- monitor.py: Main WorkflowMonitorTUI class and LayoutMode enum

Future refactoring notes:
The monitor.py could be further split into MVC components:
- model.py: State and data management
- view.py: Rich console rendering (panel creation methods)
- controller.py: Keyboard and refresh handling
"""

# Re-export public API for backward compatibility
from snakesee.tui.monitor import DEFAULT_REFRESH_RATE
from snakesee.tui.monitor import FG_BLUE
from snakesee.tui.monitor import FG_GREEN
from snakesee.tui.monitor import MAX_REFRESH_RATE
from snakesee.tui.monitor import MIN_REFRESH_RATE
from snakesee.tui.monitor import LayoutMode
from snakesee.tui.monitor import WorkflowMonitorTUI

__all__ = [
    "DEFAULT_REFRESH_RATE",
    "FG_BLUE",
    "FG_GREEN",
    "MAX_REFRESH_RATE",
    "MIN_REFRESH_RATE",
    "LayoutMode",
    "WorkflowMonitorTUI",
]
