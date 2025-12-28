"""Time estimation for Snakemake workflow progress.

This module re-exports the TimeEstimator class from the estimation package
for backward compatibility.
"""

from snakesee.estimation import TimeEstimator
from snakesee.estimation import _levenshtein_distance

__all__ = [
    "TimeEstimator",
    "_levenshtein_distance",
]
