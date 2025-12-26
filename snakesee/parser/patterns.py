"""Regex patterns for parsing Snakemake log files.

This module is the single source of truth for all log parsing patterns.
Both core.py and line_parser.py import from here to avoid duplication.
"""

import re

# Pattern: "15 of 50 steps (30%) done"
PROGRESS_PATTERN = re.compile(r"(\d+) of (\d+) steps \((\d+(?:\.\d+)?)%\) done")

# Pattern for rule start: "rule align:" or "localrule all:"
RULE_START_PATTERN = re.compile(r"(?:local)?rule (\w+):")

# Pattern for job ID in log: "    jobid: 5"
JOBID_PATTERN = re.compile(r"\s+jobid:\s*(\d+)")

# Pattern for finished job:
# - Old format: "Finished job 5." or "[date] Finished job 5."
# - Snakemake 9.x format: "Finished jobid: 5" or "Finished jobid: 5 (Rule: name)"
FINISHED_JOB_PATTERN = re.compile(r"Finished (?:job |jobid:\s*)(\d+)")

# Pattern for error in job: "Error in rule X" or job failure indicators
ERROR_PATTERN = re.compile(r"Error in rule (\w+):|Error executing rule|RuleException")

# Pattern for "Error in rule X:" specifically (to capture rule name)
ERROR_IN_RULE_PATTERN = re.compile(r"Error in rule (\w+):")

# Pattern for timestamp lines: "[Mon Dec 15 22:34:30 2025]"
TIMESTAMP_PATTERN = re.compile(r"\[(\w{3} \w{3} +\d+ \d+:\d+:\d+ \d+)\]")

# Pattern for wildcards line: "    wildcards: sample=A, batch=1"
WILDCARDS_PATTERN = re.compile(r"\s+wildcards:\s*(.+)")

# Pattern for threads line: "    threads: 4"
THREADS_PATTERN = re.compile(r"\s+threads:\s*(\d+)")

# Pattern for log line: "    log: logs/sample.log"
LOG_PATTERN = re.compile(r"\s+log:\s*(.+)")
