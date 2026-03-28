"""SQLite persistence backend.

Reads Snakemake metadata from the .snakemake/metadata.db SQLite database
introduced as an alternative persistence backend in Snakemake.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from snakesee.parser.metadata import MetadataRecord
from snakesee.parser.metadata import calculate_metadata_input_size
from snakesee.persistence.backend import IncompleteJob

if TYPE_CHECKING:
    from snakesee.state.paths import WorkflowPaths
    from snakesee.types import ProgressCallback

logger = logging.getLogger(__name__)


class DbPersistence:
    """SQLite-based persistence backend.

    Reads job metadata, incomplete markers, and lock state from
    Snakemake's .snakemake/metadata.db SQLite database.

    Opens the database read-only to avoid interfering with Snakemake.
    """

    def __init__(self, paths: WorkflowPaths) -> None:
        self._paths = paths
        self._namespace = str(paths.snakemake_dir.resolve())

    def _connect(self) -> sqlite3.Connection | None:
        """Open a read-only connection to the metadata DB.

        Returns:
            A sqlite3 Connection, or None if the DB doesn't exist or is invalid.
        """
        db_path = self._paths.metadata_db
        if not db_path.exists():
            return None

        try:
            uri = f"{db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=10.0)
            conn.row_factory = sqlite3.Row
            return conn
        except (sqlite3.DatabaseError, OSError) as e:
            logger.debug("Failed to open metadata DB %s: %s", db_path, e)
            return None

    def iterate_metadata(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> Iterator[MetadataRecord]:
        """Iterate over completed job metadata from the SQLite DB.

        Skips rows with incomplete=1, NULL rule, or stub records
        (record_format_version=0).

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            MetadataRecord for each completed job.
        """
        conn = self._connect()
        if conn is None:
            return

        try:
            if progress_callback is not None:
                row = conn.execute(
                    """SELECT COUNT(*) FROM snakemake_metadata
                       WHERE namespace = ?
                       AND (incomplete IS NULL OR incomplete = 0)
                       AND rule IS NOT NULL
                       AND record_format_version > 0""",
                    (self._namespace,),
                ).fetchone()
                total = row[0] if row else 0
            else:
                total = 0

            cursor = conn.execute(
                """SELECT rule, starttime, endtime, code, input
                   FROM snakemake_metadata
                   WHERE namespace = ?
                   AND (incomplete IS NULL OR incomplete = 0)
                   AND rule IS NOT NULL
                   AND record_format_version > 0""",
                (self._namespace,),
            )

            current = 0
            for row in cursor:
                record = self._row_to_metadata_record(row)
                if record is not None:
                    yield record

                current += 1
                if progress_callback is not None:
                    progress_callback(current, total)

        except sqlite3.DatabaseError as e:
            logger.warning("Error reading metadata from DB: %s", e)
        finally:
            conn.close()

    def has_locks(self) -> bool:
        """Check for lock rows matching our namespace.

        Returns:
            True if any locks exist for this namespace.
        """
        conn = self._connect()
        if conn is None:
            return False

        try:
            row = conn.execute(
                "SELECT 1 FROM snakemake_locks WHERE namespace = ? LIMIT 1",
                (self._namespace,),
            ).fetchone()
            return row is not None
        except sqlite3.DatabaseError as e:
            logger.debug("Error checking locks in DB: %s", e)
            return False
        finally:
            conn.close()

    def has_incomplete_jobs(self) -> bool:
        """Check for incomplete=1 rows matching our namespace.

        Returns:
            True if any incomplete jobs exist for this namespace.
        """
        conn = self._connect()
        if conn is None:
            return False

        try:
            row = conn.execute(
                """SELECT 1 FROM snakemake_metadata
                   WHERE namespace = ? AND incomplete = 1 LIMIT 1""",
                (self._namespace,),
            ).fetchone()
            return row is not None
        except sqlite3.DatabaseError as e:
            logger.debug("Error checking incomplete jobs in DB: %s", e)
            return False
        finally:
            conn.close()

    def iterate_incomplete_jobs(
        self,
        min_start_time: float | None = None,
    ) -> Iterator[IncompleteJob]:
        """Iterate over incomplete job rows from the DB.

        Args:
            min_start_time: If set, only yield jobs started at or after this time.

        Yields:
            IncompleteJob for each in-progress job.
        """
        conn = self._connect()
        if conn is None:
            return

        try:
            if min_start_time is not None:
                cursor = conn.execute(
                    """SELECT target, rule, starttime, external_jobid
                       FROM snakemake_metadata
                       WHERE namespace = ? AND incomplete = 1
                       AND starttime >= ?""",
                    (self._namespace, min_start_time),
                )
            else:
                cursor = conn.execute(
                    """SELECT target, rule, starttime, external_jobid
                       FROM snakemake_metadata
                       WHERE namespace = ? AND incomplete = 1""",
                    (self._namespace,),
                )

            for row in cursor:
                yield IncompleteJob(
                    start_time=row["starttime"],
                    output_file=Path(row["target"]) if row["target"] else None,
                    rule=row["rule"],
                    external_jobid=row["external_jobid"],
                )

        except sqlite3.DatabaseError as e:
            logger.warning("Error reading incomplete jobs from DB: %s", e)
        finally:
            conn.close()

    @staticmethod
    def _row_to_metadata_record(row: sqlite3.Row) -> MetadataRecord | None:
        """Convert a DB row to a MetadataRecord.

        Args:
            row: A sqlite3.Row with rule, starttime, endtime, code, input columns.

        Returns:
            A MetadataRecord, or None if the row has no rule.
        """
        rule = row["rule"]
        if rule is None:
            return None

        code_hash: str | None = None
        code = row["code"]
        if code:
            normalized_code = " ".join(code.split())
            code_hash = hashlib.sha256(normalized_code.encode()).hexdigest()[:16]

        input_size: int | None = None
        input_json = row["input"]
        if input_json:
            try:
                import orjson

                input_files = orjson.loads(input_json)
                if isinstance(input_files, list):
                    input_size = calculate_metadata_input_size(input_files)
            except (orjson.JSONDecodeError, TypeError) as e:
                logger.debug("Failed to parse input JSON for rule %s: %s", rule, e)

        return MetadataRecord(
            rule=rule,
            start_time=row["starttime"],
            end_time=row["endtime"],
            code_hash=code_hash,
            input_size=input_size,
        )
