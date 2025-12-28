"""Parser package for Snakemake log file parsing.

This package provides components for parsing Snakemake log files:

- IncrementalLogReader: Streaming reader with position tracking
- LogFilePosition: File position and rotation tracking
- LogLineParser: Line-by-line parsing with context
- JobLifecycleTracker: Job start/finish tracking
- FailureTracker: Failure deduplication

The parser module is split into focused components:
- metadata.py: Metadata file parsing (MetadataRecord, parse_metadata_files)
- stats.py: Timing statistics collection (collect_rule_timing_stats)
- utils.py: Utility functions (_parse_wildcards, calculate_input_size)
- core.py: Job parsing and workflow state assembly

Related modules:
    snakesee.estimation: Uses parsed timing data for ETA estimation
    snakesee.models: JobInfo and WorkflowProgress data classes
    snakesee.state.paths: WorkflowPaths for finding log files
    snakesee.validation: Validates parsed state against events
"""

# Import from focused modules
# Import job parsing and workflow state from core
from snakesee.parser.core import _augment_completions_with_threads
from snakesee.parser.core import is_workflow_running
from snakesee.parser.core import parse_all_jobs_from_log
from snakesee.parser.core import parse_completed_jobs_from_log
from snakesee.parser.core import parse_failed_jobs_from_log
from snakesee.parser.core import parse_incomplete_jobs
from snakesee.parser.core import parse_job_stats_counts_from_log
from snakesee.parser.core import parse_job_stats_from_log
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
from snakesee.parser.metadata import MetadataRecord
from snakesee.parser.metadata import collect_rule_code_hashes
from snakesee.parser.metadata import parse_metadata_files
from snakesee.parser.metadata import parse_metadata_files_full
from snakesee.parser.patterns import ERROR_IN_RULE_PATTERN
from snakesee.parser.patterns import ERROR_PATTERN
from snakesee.parser.patterns import FINISHED_JOB_PATTERN
from snakesee.parser.patterns import JOBID_PATTERN
from snakesee.parser.patterns import LOG_PATTERN
from snakesee.parser.patterns import PROGRESS_PATTERN
from snakesee.parser.patterns import RULE_START_PATTERN
from snakesee.parser.patterns import THREADS_PATTERN
from snakesee.parser.patterns import TIMESTAMP_PATTERN
from snakesee.parser.patterns import WILDCARDS_PATTERN
from snakesee.parser.stats import collect_rule_timing_stats
from snakesee.parser.stats import collect_wildcard_timing_stats
from snakesee.parser.utils import _parse_non_negative_int
from snakesee.parser.utils import _parse_positive_int
from snakesee.parser.utils import _parse_timestamp
from snakesee.parser.utils import _parse_wildcards
from snakesee.parser.utils import calculate_input_size
from snakesee.parser.utils import estimate_input_size_from_output

__all__ = [  # noqa: RUF022 (keep grouped by category for readability)
    # Classes (alphabetical)
    "FailureTracker",
    "IncrementalLogReader",
    "JobLifecycleTracker",
    "LogFilePosition",
    "LogLineParser",
    "MetadataRecord",
    "ParseEvent",
    "ParseEventType",
    "ParsingContext",
    "StartedJobData",
    # Functions (alphabetical)
    "calculate_input_size",
    "collect_rule_code_hashes",
    "collect_rule_timing_stats",
    "collect_wildcard_timing_stats",
    "estimate_input_size_from_output",
    "is_workflow_running",
    "parse_all_jobs_from_log",
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
    # Patterns (alphabetical, for advanced usage)
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
    # Private functions (alphabetical, exported for tests)
    "_augment_completions_with_threads",
    "_parse_non_negative_int",
    "_parse_positive_int",
    "_parse_timestamp",
    "_parse_wildcards",
]
