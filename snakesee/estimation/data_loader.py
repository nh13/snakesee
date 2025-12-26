"""Historical data loading for time estimation."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from snakesee.constants import MAX_EVENTS_LINE_LENGTH
from snakesee.parser import parse_metadata_files_full

if TYPE_CHECKING:
    from snakesee.state.rule_registry import RuleRegistry
    from snakesee.types import ProgressCallback

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Loads timing data from metadata and events files.

    Provides methods to load historical execution data from:
    - .snakemake/metadata/ directory (from previous Snakemake runs)
    - .snakesee_events.jsonl file (from snakesee monitoring)
    """

    def __init__(
        self,
        registry: "RuleRegistry",
        use_wildcard_conditioning: bool = False,
    ) -> None:
        """Initialize the loader.

        Args:
            registry: RuleRegistry to load data into.
            use_wildcard_conditioning: Whether to record wildcard-specific stats.
        """
        self._registry = registry
        self.use_wildcard_conditioning = use_wildcard_conditioning
        self.code_hash_to_rules: dict[str, set[str]] = {}

    def load_from_metadata(
        self,
        metadata_dir: Path,
        progress_callback: "ProgressCallback | None" = None,
    ) -> None:
        """Load historical execution times from .snakemake/metadata/.

        Uses a single-pass parser for efficiency - reads each metadata file
        only once to collect timing stats, code hashes, and wildcard stats.

        Args:
            metadata_dir: Path to .snakemake/metadata/ directory.
            progress_callback: Optional callback(current, total) for progress.
        """
        hash_to_rules: dict[str, set[str]] = {}

        for record in parse_metadata_files_full(metadata_dir, progress_callback):
            duration = record.duration
            end_time = record.end_time

            if duration is not None and end_time is not None:
                wildcards = record.wildcards if self.use_wildcard_conditioning else None
                self._registry.record_completion(
                    rule=record.rule,
                    duration=duration,
                    timestamp=end_time,
                    wildcards=wildcards,
                    input_size=record.input_size,
                )

            if record.code_hash:
                if record.code_hash not in hash_to_rules:
                    hash_to_rules[record.code_hash] = set()
                hash_to_rules[record.code_hash].add(record.rule)

        self.code_hash_to_rules = hash_to_rules

    def load_from_events(self, events_file: Path) -> bool:
        """Load historical execution times from a snakesee events file.

        Streams the events file line by line for memory efficiency.

        Args:
            events_file: Path to .snakesee_events.jsonl file.

        Returns:
            True if any wildcard data was found.
        """
        if not events_file.exists():
            return False

        has_wildcards = False

        try:
            with open(events_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    # Skip excessively long lines to prevent memory issues
                    if len(line) > MAX_EVENTS_LINE_LENGTH:
                        logger.debug(
                            "Skipping oversized line in events file: %d bytes (max %d)",
                            len(line),
                            MAX_EVENTS_LINE_LENGTH,
                        )
                        continue

                    try:
                        event = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue

                    if event.get("event_type") != "job_finished":
                        continue

                    duration = event.get("duration")
                    timestamp = event.get("timestamp")
                    rule_name = event.get("rule_name")
                    wildcards = event.get("wildcards")

                    if duration is None or timestamp is None or rule_name is None:
                        continue

                    wc_dict = wildcards if isinstance(wildcards, dict) else None
                    self._registry.record_completion(
                        rule=rule_name,
                        duration=duration,
                        timestamp=timestamp,
                        wildcards=wc_dict,
                    )

                    if wc_dict:
                        has_wildcards = True

        except OSError as e:
            logger.warning("Error reading events file %s: %s", events_file, e)
            return False

        return has_wildcards
