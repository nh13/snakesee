"""Centralized constants for snakesee.

This module consolidates configuration constants and magic numbers
used across multiple modules to ensure consistency and make tuning easier.

Configuration is organized into two modules:

1. This module (constants.py): Runtime configuration
   - RefreshRateConfig: TUI refresh rates
   - CacheConfig: Caching behavior and TTLs
   - FileSizeLimits: Security limits for file parsing

2. snakesee.state.config: Estimation-specific configuration
   - EstimationConfig: Time estimation parameters
   - VarianceMultipliers: Variance settings per estimation method
   - ConfidenceWeights: Confidence calculation weights
   - ConfidenceThresholds: Decision thresholds
   - TimeConstants: Time-related constants

Note: This module defines STALE_WORKFLOW_THRESHOLD_SECONDS (1800.0 seconds / 30 minutes),
which is imported by state.config.TimeConstants.stale_workflow_threshold to ensure
a single source of truth.

For estimation-specific configuration, see :class:`snakesee.state.config.EstimationConfig`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RefreshRateConfig:
    """Configuration for TUI refresh rate.

    Attributes:
        min_rate: Minimum refresh rate in seconds (fastest).
        max_rate: Maximum refresh rate in seconds (slowest).
        default_rate: Default refresh rate in seconds.
    """

    min_rate: float = 0.5
    max_rate: float = 60.0
    default_rate: float = 1.0


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for caching behavior.

    Attributes:
        default_tool_progress_ttl: Default TTL for tool progress cache in seconds.
        adaptive_ttl_multiplier: Multiplier for adaptive cache TTL based on refresh rate.
        max_cache_ttl: Maximum cache TTL in seconds (cap for adaptive calculation).
        exists_cache_ttl: TTL for filesystem existence check cache in seconds.
    """

    default_tool_progress_ttl: float = 5.0
    adaptive_ttl_multiplier: float = 2.5
    max_cache_ttl: float = 15.0
    exists_cache_ttl: float = 5.0


@dataclass(frozen=True)
class FileSizeLimits:
    """Security limits for file sizes.

    Attributes:
        max_metadata_file_size: Maximum size in bytes for metadata files (10 MB).
        max_events_line_length: Maximum line length in bytes for events file (1 MB).
    """

    max_metadata_file_size: int = 10 * 1024 * 1024
    max_events_line_length: int = 1 * 1024 * 1024


# Default instances for easy access
REFRESH_RATE_CONFIG = RefreshRateConfig()
CACHE_CONFIG = CacheConfig()
FILE_SIZE_LIMITS = FileSizeLimits()

# =============================================================================
# Module-level constants (derived from config classes above)
# =============================================================================

#: Minimum refresh rate in seconds (fastest)
MIN_REFRESH_RATE: float = REFRESH_RATE_CONFIG.min_rate

#: Maximum refresh rate in seconds (slowest)
MAX_REFRESH_RATE: float = REFRESH_RATE_CONFIG.max_rate

#: Default refresh rate in seconds
DEFAULT_REFRESH_RATE: float = REFRESH_RATE_CONFIG.default_rate

#: Seconds since last log modification before considering workflow stale/dead
#: Default is 30 minutes (1800 seconds)
#: Note: TimeConstants.stale_workflow_threshold in state/config.py imports this value
STALE_WORKFLOW_THRESHOLD_SECONDS: float = 1800.0

#: Default TTL for tool progress cache in seconds
DEFAULT_TOOL_PROGRESS_CACHE_TTL: float = CACHE_CONFIG.default_tool_progress_ttl

#: Multiplier for adaptive cache TTL based on refresh rate
ADAPTIVE_CACHE_TTL_MULTIPLIER: float = CACHE_CONFIG.adaptive_ttl_multiplier

#: Maximum cache TTL in seconds (cap for adaptive calculation)
MAX_CACHE_TTL: float = CACHE_CONFIG.max_cache_ttl

#: Maximum size in bytes for metadata files before skipping (10 MB)
MAX_METADATA_FILE_SIZE: int = FILE_SIZE_LIMITS.max_metadata_file_size

#: Maximum line length in bytes for events file parsing (1 MB)
MAX_EVENTS_LINE_LENGTH: int = FILE_SIZE_LIMITS.max_events_line_length

#: TTL for filesystem existence check cache in seconds
EXISTS_CACHE_TTL: float = CACHE_CONFIG.exists_cache_ttl

#: Minimum samples required for wildcard/combination conditioning
#: Used by WildcardTimingStats and RuleRegistry for statistical validity
MIN_SAMPLES_FOR_CONDITIONING: int = 3
