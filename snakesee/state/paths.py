"""Centralized path management for Snakemake workflow directories.

This module provides the WorkflowPaths class which centralizes all path
construction for Snakemake workflows, eliminating ad-hoc path construction
scattered across multiple functions.
"""

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from snakesee.constants import EXISTS_CACHE_TTL
from snakesee.utils import safe_mtime

# Module-level cache for filesystem existence checks
# Maps path string -> (timestamp, exists_result)
_exists_cache: dict[str, tuple[float, bool]] = {}


def _cached_exists(path: Path, ttl: float = EXISTS_CACHE_TTL) -> bool:
    """Check if a path exists, with caching to reduce filesystem calls.

    This is particularly beneficial on network filesystems where stat() calls
    can be slow.

    Args:
        path: Path to check.
        ttl: Cache TTL in seconds (default from constants).

    Returns:
        True if the path exists, False otherwise.
    """
    key = str(path)
    now = time.time()

    if key in _exists_cache:
        cached_time, cached_result = _exists_cache[key]
        if now - cached_time < ttl:
            return cached_result

    result = path.exists()
    _exists_cache[key] = (now, result)
    return result


def clear_exists_cache() -> None:
    """Clear the filesystem existence cache.

    Useful for testing or after significant filesystem operations.
    """
    _exists_cache.clear()


# Default names for snakemake directories and files
SNAKEMAKE_DIR = ".snakemake"
METADATA_DIR = "metadata"
LOG_DIR = "log"
INCOMPLETE_DIR = "incomplete"
LOCKS_DIR = "locks"
LOG_GLOB_PATTERN = "*.snakemake.log"
EVENT_FILE_NAME = ".snakesee_events.jsonl"
VALIDATION_LOG_NAME = ".snakesee_validation.log"
DEFAULT_PROFILE_NAME = ".snakesee-profile.json"


@dataclass(frozen=True)
class WorkflowPaths:
    """Centralized path management for Snakemake workflow directories.

    This frozen dataclass provides computed paths for all standard
    Snakemake directory locations, eliminating ad-hoc path construction.

    Attributes:
        workflow_dir: Root directory of the workflow (contains .snakemake/).

    Example:
        paths = WorkflowPaths(Path("/my/workflow"))

        # Access computed paths
        if paths.metadata_dir.exists():
            for f in paths.get_metadata_files():
                process(f)

        # Find logs
        latest = paths.find_latest_log()
        all_logs = paths.find_all_logs()
    """

    workflow_dir: Path

    # =========================================================================
    # Core directory properties
    # =========================================================================

    @property
    def snakemake_dir(self) -> Path:
        """Path to .snakemake/ directory."""
        return self.workflow_dir / SNAKEMAKE_DIR

    @property
    def metadata_dir(self) -> Path:
        """Path to .snakemake/metadata/ directory."""
        return self.snakemake_dir / METADATA_DIR

    @property
    def log_dir(self) -> Path:
        """Path to .snakemake/log/ directory."""
        return self.snakemake_dir / LOG_DIR

    @property
    def incomplete_dir(self) -> Path:
        """Path to .snakemake/incomplete/ directory."""
        return self.snakemake_dir / INCOMPLETE_DIR

    @property
    def locks_dir(self) -> Path:
        """Path to .snakemake/locks/ directory."""
        return self.snakemake_dir / LOCKS_DIR

    # =========================================================================
    # Event and validation file paths
    # =========================================================================

    @property
    def events_file(self) -> Path:
        """Path to snakesee events file (.snakesee_events.jsonl)."""
        return self.workflow_dir / EVENT_FILE_NAME

    @property
    def validation_log(self) -> Path:
        """Path to validation log file (.snakesee_validation.log)."""
        return self.workflow_dir / VALIDATION_LOG_NAME

    @property
    def default_profile(self) -> Path:
        """Path to default profile file (.snakesee-profile.json)."""
        return self.workflow_dir / DEFAULT_PROFILE_NAME

    # =========================================================================
    # Existence checks
    # =========================================================================

    @property
    def exists(self) -> bool:
        """Check if this is a valid workflow directory."""
        return _cached_exists(self.snakemake_dir)

    @property
    def has_metadata(self) -> bool:
        """Check if metadata directory exists."""
        return _cached_exists(self.metadata_dir)

    @property
    def has_logs(self) -> bool:
        """Check if log directory exists and contains logs."""
        if not _cached_exists(self.log_dir):
            return False
        return any(self.log_dir.glob(LOG_GLOB_PATTERN))

    @property
    def has_events(self) -> bool:
        """Check if events file exists and has content."""
        try:
            return self.events_file.stat().st_size > 0
        except OSError:
            return False

    @property
    def has_locks(self) -> bool:
        """Check if locks directory exists and contains files."""
        if not _cached_exists(self.locks_dir):
            return False
        try:
            return any(self.locks_dir.iterdir())
        except OSError:
            return False

    @property
    def has_incomplete(self) -> bool:
        """Check if incomplete directory exists and contains markers."""
        if not _cached_exists(self.incomplete_dir):
            return False
        try:
            return any(self.incomplete_dir.iterdir())
        except OSError:
            return False

    # =========================================================================
    # Log file discovery
    # =========================================================================

    def find_latest_log(self) -> Path | None:
        """Find the most recent snakemake log file.

        Returns:
            Path to the most recent log file, or None if no logs exist.
        """
        if not _cached_exists(self.log_dir):
            return None
        # Files from glob already exist at time of iteration; no need to re-check
        logs = list(self.log_dir.glob(LOG_GLOB_PATTERN))
        if not logs:
            return None
        logs.sort(key=safe_mtime)
        return logs[-1]

    def find_all_logs(self) -> list[Path]:
        """Find all snakemake log files, sorted by modification time.

        Returns:
            List of paths sorted oldest to newest.
        """
        if not _cached_exists(self.log_dir):
            return []
        # Files from glob already exist at time of iteration; no need to re-check
        logs = list(self.log_dir.glob(LOG_GLOB_PATTERN))
        logs.sort(key=safe_mtime)
        return logs

    def find_logs_sorted_newest_first(self) -> list[Path]:
        """Find all snakemake log files, sorted newest first.

        Returns:
            List of paths sorted newest to oldest.
        """
        logs = self.find_all_logs()
        logs.reverse()
        return logs

    # =========================================================================
    # Metadata file discovery
    # =========================================================================

    def get_metadata_files(self) -> Iterator[Path]:
        """Iterate over all metadata files.

        Yields:
            Path to each metadata file.
        """
        if not _cached_exists(self.metadata_dir):
            return
        for f in self.metadata_dir.rglob("*"):
            if f.is_file():
                yield f

    def count_metadata_files(self) -> int:
        """Count the number of metadata files.

        Returns:
            Number of metadata files.
        """
        if not _cached_exists(self.metadata_dir):
            return 0
        return sum(1 for f in self.metadata_dir.rglob("*") if f.is_file())

    # =========================================================================
    # Incomplete marker handling
    # =========================================================================

    def get_incomplete_markers(self) -> Iterator[Path]:
        """Iterate over incomplete job markers.

        Yields:
            Path to each incomplete marker file.
        """
        if not _cached_exists(self.incomplete_dir):
            return
        for marker in self.incomplete_dir.rglob("*"):
            if marker.is_file() and marker.name != "migration_underway":
                yield marker

    def decode_incomplete_marker(self, marker: Path) -> Path | None:
        """Decode an incomplete marker filename to get the output path.

        Args:
            marker: Path to the marker file.

        Returns:
            Decoded output file path, or None if decoding fails.
        """
        try:
            decoded = base64.b64decode(marker.name).decode("utf-8")
            return Path(decoded)
        except (ValueError, UnicodeDecodeError):
            return None

    # =========================================================================
    # Job log discovery
    # =========================================================================

    def get_job_log(
        self,
        rule: str,
        wildcards: dict[str, str] | None = None,
        job_id: int | str | None = None,
    ) -> Path | None:
        """Find the log file for a specific job.

        Searches common log locations for a file matching the rule
        and optional wildcards/job_id.

        Args:
            rule: Name of the rule.
            wildcards: Optional wildcard values.
            job_id: Optional job ID.

        Returns:
            Path to the log file if found, None otherwise.
        """
        search_paths: list[Path] = []

        # .snakemake/log/ directory
        if _cached_exists(self.log_dir):
            search_paths.extend(self.log_dir.glob(f"*{rule}*"))
            if job_id is not None:
                search_paths.extend(self.log_dir.glob(f"*job{job_id}*"))

        # logs/ directory (common convention)
        logs_dir = self.workflow_dir / "logs"
        search_paths.extend(self._search_log_dir(logs_dir, rule, wildcards))

        # log/ directory (another common convention)
        log_dir = self.workflow_dir / "log"
        search_paths.extend(self._search_log_dir(log_dir, rule, wildcards))

        # Sort by modification time (newest first) and return first match
        # is_file() already confirms existence, no need for additional exists check
        existing_logs = [p for p in search_paths if p.is_file()]
        if existing_logs:
            existing_logs.sort(key=safe_mtime, reverse=True)
            return existing_logs[0]

        return None

    def _search_log_dir(
        self,
        log_dir: Path,
        rule: str,
        wildcards: dict[str, str] | None,
    ) -> list[Path]:
        """Search a log directory for matching logs."""
        paths: list[Path] = []
        if not _cached_exists(log_dir):
            return paths

        paths.extend(log_dir.glob(f"**/{rule}*"))

        rule_log_dir = log_dir / rule
        if _cached_exists(rule_log_dir):
            paths.extend(rule_log_dir.glob("*"))

        if wildcards:
            for wc_value in wildcards.values():
                if wc_value:
                    paths.extend(log_dir.glob(f"**/*{wc_value}*"))

        return paths

    # =========================================================================
    # Profile discovery
    # =========================================================================

    def find_profile(self, max_levels: int = 6) -> Path | None:
        """Search for a profile file in workflow and parent directories.

        Args:
            max_levels: Maximum parent levels to search (including current).

        Returns:
            Path to the found profile, or None if not found.
        """
        current = self.workflow_dir.resolve()
        for _ in range(max_levels):
            profile_path = current / DEFAULT_PROFILE_NAME
            if _cached_exists(profile_path):
                return profile_path
            if current.parent == current:
                break
            current = current.parent
        return None

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> None:
        """Validate that this is a valid workflow directory.

        Raises:
            ValueError: If .snakemake directory doesn't exist.
        """
        if not _cached_exists(self.snakemake_dir):
            raise ValueError(f"No .snakemake directory found in {self.workflow_dir}")
