"""Time estimation package for Snakemake workflow progress.

This package provides modular components for estimating remaining workflow time:
- TimeEstimator: Main coordinator class
- RuleMatchingEngine: Fuzzy rule matching by name/code hash
- HistoricalDataLoader: Load timing data from metadata/events
- PendingRuleInferrer: Infer pending job distribution

Related modules:
    snakesee.parser: Parses log files and metadata to extract timing data
    snakesee.models: Data classes for JobInfo, WorkflowProgress, TimeEstimate
    snakesee.state: RuleRegistry for storing timing statistics
    snakesee.formatting: Format durations and ETAs for display
"""

from snakesee.estimation.data_loader import HistoricalDataLoader
from snakesee.estimation.estimator import TimeEstimator
from snakesee.estimation.pending_inferrer import PendingRuleInferrer
from snakesee.estimation.rule_matcher import RuleMatchingEngine
from snakesee.estimation.rule_matcher import levenshtein_distance

# Backward compatibility alias
_levenshtein_distance = levenshtein_distance

__all__ = [
    "TimeEstimator",
    "RuleMatchingEngine",
    "HistoricalDataLoader",
    "PendingRuleInferrer",
    "levenshtein_distance",
    "_levenshtein_distance",
]
