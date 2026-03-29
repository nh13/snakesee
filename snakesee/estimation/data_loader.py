"""Historical data loading for time estimation."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from snakesee.constants import MAX_EVENTS_LINE_LENGTH
from snakesee.parser import parse_metadata_files_full
from snakesee.parser.metadata import MetadataRecord

if TYPE_CHECKING:
    from snakesee.persistence.backend import PersistenceBackend
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
        self._consume_metadata_records(parse_metadata_files_full(metadata_dir, progress_callback))

    def load_from_backend(
        self,
        backend: "PersistenceBackend",
        progress_callback: "ProgressCallback | None" = None,
    ) -> None:
        """Load historical execution times from a persistence backend.

        Works with both filesystem and SQLite backends via the
        PersistenceBackend protocol.

        Args:
            backend: Persistence backend to read from.
            progress_callback: Optional callback(current, total) for progress.
        """
        self._consume_metadata_records(backend.iterate_metadata(progress_callback))

    def _consume_metadata_records(
        self,
        records: "Iterator[MetadataRecord]",
    ) -> None:
        """Process metadata records into the registry and code hash map.

        Args:
            records: Iterator of MetadataRecord instances.
        """
        hash_to_rules: dict[str, set[str]] = {}

        for record in records:
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
                    threads = event.get("threads")

                    if duration is None or timestamp is None or rule_name is None:
                        continue

                    wc_dict = wildcards if isinstance(wildcards, dict) else None
                    threads_int = None
                    if threads is not None:
                        try:
                            candidate = int(threads)
                        except (TypeError, ValueError):
                            logger.debug(
                                "Ignoring invalid thread count in events file: %r",
                                threads,
                            )
                        else:
                            if candidate > 0:
                                threads_int = candidate
                    self._registry.record_completion(
                        rule=rule_name,
                        duration=duration,
                        timestamp=timestamp,
                        wildcards=wc_dict,
                        threads=threads_int,
                    )

                    if wc_dict:
                        has_wildcards = True

        except OSError as e:
            logger.warning("Error reading events file %s: %s", events_file, e)
            return False

        return has_wildcards
