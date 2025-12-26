"""Centralized constants for snakesee.

This module consolidates configuration constants and magic numbers
used across multiple modules to ensure consistency and make tuning easier.
"""

# =============================================================================
# Refresh Rate Configuration
# =============================================================================

#: Minimum refresh rate in seconds (fastest)
MIN_REFRESH_RATE: float = 0.5

#: Maximum refresh rate in seconds (slowest)
MAX_REFRESH_RATE: float = 60.0

#: Default refresh rate in seconds
DEFAULT_REFRESH_RATE: float = 1.0

# =============================================================================
# Workflow State Detection
# =============================================================================

#: Seconds since last log modification before considering workflow stale/dead
#: Default is 30 minutes (1800 seconds)
STALE_WORKFLOW_THRESHOLD_SECONDS: float = 1800.0

# =============================================================================
# Tool Progress Cache
# =============================================================================

#: Default TTL for tool progress cache in seconds
DEFAULT_TOOL_PROGRESS_CACHE_TTL: float = 5.0

#: Multiplier for adaptive cache TTL based on refresh rate
#: cache_ttl = min(ADAPTIVE_CACHE_TTL_MULTIPLIER * refresh_rate, MAX_CACHE_TTL)
ADAPTIVE_CACHE_TTL_MULTIPLIER: float = 2.5

#: Maximum cache TTL in seconds (cap for adaptive calculation)
MAX_CACHE_TTL: float = 15.0

# =============================================================================
# File Size Limits
# =============================================================================

#: Maximum size in bytes for metadata files before skipping (10 MB)
#: Prevents DoS from maliciously large files
MAX_METADATA_FILE_SIZE: int = 10 * 1024 * 1024

#: Maximum line length in bytes for events file parsing (1 MB)
#: Prevents memory issues from malformed lines
MAX_EVENTS_LINE_LENGTH: int = 1 * 1024 * 1024

# =============================================================================
# Filesystem Caching
# =============================================================================

#: TTL for filesystem existence check cache in seconds
#: Reduces repeated stat() calls on the same paths
EXISTS_CACHE_TTL: float = 5.0
