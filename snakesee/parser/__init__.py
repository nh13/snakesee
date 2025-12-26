"""Parser package for Snakemake log file parsing.

This package provides components for parsing Snakemake log files:

- IncrementalLogReader: Streaming reader with position tracking
- LogFilePosition: File position and rotation tracking
- LogLineParser: Line-by-line parsing with context
- JobLifecycleTracker: Job start/finish tracking
- FailureTracker: Failure deduplication
"""

# Re-export new modular components
from snakesee.parser.core import ERROR_IN_RULE_PATTERN
from snakesee.parser.core import ERROR_PATTERN
from snakesee.parser.core import FINISHED_JOB_PATTERN
from snakesee.parser.core import JOBID_PATTERN
from snakesee.parser.core import LOG_PATTERN
from snakesee.parser.core import PROGRESS_PATTERN
from snakesee.parser.core import RULE_START_PATTERN
from snakesee.parser.core import THREADS_PATTERN
from snakesee.parser.core import TIMESTAMP_PATTERN
from snakesee.parser.core import WILDCARDS_PATTERN
from snakesee.parser.core import MetadataRecord
from snakesee.parser.core import _augment_completions_with_threads
from snakesee.parser.core import _parse_non_negative_int
from snakesee.parser.core import _parse_positive_int
from snakesee.parser.core import _parse_timestamp
from snakesee.parser.core import _parse_wildcards

# Re-export all public API from core module for backward compatibility
from snakesee.parser.core import calculate_input_size
from snakesee.parser.core import collect_rule_code_hashes
from snakesee.parser.core import collect_rule_timing_stats
from snakesee.parser.core import collect_wildcard_timing_stats
from snakesee.parser.core import estimate_input_size_from_output
from snakesee.parser.core import is_workflow_running
from snakesee.parser.core import parse_completed_jobs_from_log
from snakesee.parser.core import parse_failed_jobs_from_log
from snakesee.parser.core import parse_incomplete_jobs
from snakesee.parser.core import parse_job_stats_counts_from_log
from snakesee.parser.core import parse_job_stats_from_log
from snakesee.parser.core import parse_metadata_files
from snakesee.parser.core import parse_metadata_files_full
from snakesee.parser.core import parse_progress_from_log
from snakesee.parser.core import parse_rules_from_log
from snakesee.parser.core import parse_running_jobs_from_log
from snakesee.parser.core import parse_threads_from_log
from snakesee.parser.core import parse_workflow_state
from snakesee.parser.failure_tracker import FailureTracker
from snakesee.parser.file_position import LogFilePosition
from snakesee.parser.job_tracker import JobLifecycleTracker
from snakesee.parser.job_tracker import StartedJobData
from snakesee.parser.line_parser import LogLineParser
from snakesee.parser.line_parser import ParseEvent
from snakesee.parser.line_parser import ParseEventType
from snakesee.parser.line_parser import ParsingContext
from snakesee.parser.log_reader import IncrementalLogReader

__all__ = [
    # New modular components
    "IncrementalLogReader",
    "LogFilePosition",
    "LogLineParser",
    "JobLifecycleTracker",
    "FailureTracker",
    "ParseEvent",
    "ParseEventType",
    "ParsingContext",
    "StartedJobData",
    # Core parsing functions
    "calculate_input_size",
    "collect_rule_code_hashes",
    "collect_rule_timing_stats",
    "collect_wildcard_timing_stats",
    "estimate_input_size_from_output",
    "is_workflow_running",
    "parse_completed_jobs_from_log",
    "parse_failed_jobs_from_log",
    "parse_incomplete_jobs",
    "parse_job_stats_counts_from_log",
    "parse_job_stats_from_log",
    "parse_metadata_files",
    "parse_metadata_files_full",
    "parse_progress_from_log",
    "parse_rules_from_log",
    "parse_running_jobs_from_log",
    "parse_threads_from_log",
    "parse_workflow_state",
    # Patterns (for advanced usage)
    "ERROR_IN_RULE_PATTERN",
    "ERROR_PATTERN",
    "FINISHED_JOB_PATTERN",
    "JOBID_PATTERN",
    "LOG_PATTERN",
    "PROGRESS_PATTERN",
    "RULE_START_PATTERN",
    "THREADS_PATTERN",
    "TIMESTAMP_PATTERN",
    "WILDCARDS_PATTERN",
    # Types
    "MetadataRecord",
    # Private functions (exported for tests)
    "_augment_completions_with_threads",
    "_parse_non_negative_int",
    "_parse_positive_int",
    "_parse_timestamp",
    "_parse_wildcards",
]
