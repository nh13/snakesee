"""Snakesee: A terminal UI for monitoring Snakemake workflows."""

from importlib.metadata import version
from pathlib import Path

from snakesee.estimator import TimeEstimator
from snakesee.events import EVENT_FILE_NAME
from snakesee.events import EventReader
from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.events import get_event_file_path
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
    "EVENT_FILE_NAME",
    "EventReader",
    "EventType",
    "JobInfo",
    "LOG_HANDLER_SCRIPT",
    "RuleTimingStats",
    "SnakeseeEvent",
    "TimeEstimate",
    "TimeEstimator",
    "WorkflowProgress",
    "WorkflowStatus",
    "format_duration",
    "get_event_file_path",
    "parse_workflow_state",
]
