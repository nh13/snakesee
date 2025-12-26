"""Snakesee: A terminal UI for monitoring Snakemake workflows."""

from importlib.metadata import version
from pathlib import Path

from snakesee.estimator import TimeEstimator
from snakesee.events import EVENT_FILE_NAME
from snakesee.events import EventReader
from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.events import get_event_file_path
from snakesee.exceptions import ConfigurationError
from snakesee.exceptions import InvalidProfileError
from snakesee.exceptions import PluginError
from snakesee.exceptions import PluginExecutionError
from snakesee.exceptions import PluginLoadError
from snakesee.exceptions import ProfileError
from snakesee.exceptions import ProfileNotFoundError
from snakesee.exceptions import SnakeseeError
from snakesee.exceptions import WorkflowError
from snakesee.exceptions import WorkflowNotFoundError
from snakesee.exceptions import WorkflowParseError
from snakesee.models import JobInfo
from snakesee.models import RuleTimingStats
from snakesee.models import TimeEstimate
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.models import format_duration
from snakesee.parser import parse_workflow_state

__version__ = version("snakesee")

# Path to the log handler script for Snakemake 8.x --log-handler-script
LOG_HANDLER_SCRIPT = Path(__file__).parent / "log_handler_script.py"

__all__ = [
    # Exceptions
    "ConfigurationError",
    "InvalidProfileError",
    "PluginError",
    "PluginExecutionError",
    "PluginLoadError",
    "ProfileError",
    "ProfileNotFoundError",
    "SnakeseeError",
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowParseError",
    # Events
    "EVENT_FILE_NAME",
    "EventReader",
    "EventType",
    "SnakeseeEvent",
    "get_event_file_path",
    # Models
    "JobInfo",
    "RuleTimingStats",
    "TimeEstimate",
    "WorkflowProgress",
    "WorkflowStatus",
    # Functions
    "format_duration",
    "parse_workflow_state",
    # Components
    "LOG_HANDLER_SCRIPT",
    "TimeEstimator",
]
