"""Tests for SQLite persistence backend."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import pytest

from snakesee.persistence.db import DbPersistence
from snakesee.state.paths import WorkflowPaths
from snakesee.state.paths import clear_exists_cache
from tests.db_helpers import create_metadata_schema


def create_test_db(tmp_path: Path) -> tuple[Path, str]:
    """Create a test metadata.db and return its path and namespace."""
    smk_dir = tmp_path / ".snakemake"
    smk_dir.mkdir(parents=True, exist_ok=True)
    db_path = smk_dir / "metadata.db"
    namespace = str(smk_dir.resolve())
    conn = sqlite3.connect(str(db_path))
    create_metadata_schema(conn)
    conn.commit()
    conn.close()
    return db_path, namespace


def insert_metadata(
    db_path: Path,
    namespace: str,
    target: str,
    rule: str | None,
    starttime: float | None = None,
    endtime: float | None = None,
    code: str | None = None,
    incomplete: bool = False,
    external_jobid: str | None = None,
    input_json: str | None = None,
    record_format_version: int = 6,
) -> None:
    """Insert a row into snakemake_metadata."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT INTO snakemake_metadata
           (namespace, target, rule, starttime, endtime, code,
            incomplete, external_jobid, input, record_format_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            namespace,
            target,
            rule,
            starttime,
            endtime,
            code,
            1 if incomplete else 0,
            external_jobid,
            input_json,
            record_format_version,
        ),
    )
    conn.commit()
    conn.close()


def insert_lock(
    db_path: Path,
    namespace: str,
    file_path: str,
    lock_type: str,
) -> None:
    """Insert a row into snakemake_locks."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO snakemake_locks (namespace, file_path, lock_type) VALUES (?, ?, ?)",
        (namespace, file_path, lock_type),
    )
    conn.commit()
    conn.close()


def make_backend(tmp_path: Path) -> DbPersistence:
    """Create a DbPersistence instance for the given tmp_path."""
    clear_exists_cache()
    paths = WorkflowPaths(tmp_path)
    return DbPersistence(paths)


# =============================================================================
# Metadata tests
# =============================================================================


class TestDbPersistenceMetadata:
    """Tests for iterate_metadata."""

    def test_iterate_empty_db(self, tmp_path: Path) -> None:
        """An empty DB yields no records."""
        create_test_db(tmp_path)
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())
        assert records == []

    def test_iterate_metadata_returns_records(self, tmp_path: Path) -> None:
        """Completed rows are returned as MetadataRecords with correct fields."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="output.txt",
            rule="align",
            starttime=1000.0,
            endtime=1060.0,
            code="bwa mem {input} > {output}",
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())

        assert len(records) == 1
        rec = records[0]
        assert rec.rule == "align"
        assert rec.start_time == 1000.0
        assert rec.end_time == 1060.0
        assert rec.duration == 60.0
        assert rec.code_hash is not None
        assert len(rec.code_hash) == 16

    def test_iterate_metadata_skips_incomplete(self, tmp_path: Path) -> None:
        """Rows with incomplete=1 are skipped."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="running.txt",
            rule="slow_rule",
            starttime=1000.0,
            incomplete=True,
        )
        insert_metadata(
            db_path,
            namespace,
            target="done.txt",
            rule="fast_rule",
            starttime=900.0,
            endtime=910.0,
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())

        assert len(records) == 1
        assert records[0].rule == "fast_rule"

    def test_iterate_metadata_skips_null_rule(self, tmp_path: Path) -> None:
        """Rows with NULL rule are skipped."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="no_rule.txt",
            rule=None,
            starttime=1000.0,
            endtime=1010.0,
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())
        assert records == []

    def test_iterate_metadata_skips_stub_records(self, tmp_path: Path) -> None:
        """Rows with record_format_version=0 (stubs) are skipped."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="stub.txt",
            rule="stub_rule",
            starttime=1000.0,
            endtime=1010.0,
            record_format_version=0,
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())
        assert records == []

    def test_iterate_metadata_filters_by_namespace(self, tmp_path: Path) -> None:
        """Rows with a different namespace are ignored."""
        db_path, namespace = create_test_db(tmp_path)
        # Insert a row with our namespace
        insert_metadata(
            db_path,
            namespace,
            target="ours.txt",
            rule="our_rule",
            starttime=1000.0,
            endtime=1010.0,
        )
        # Insert a row with a foreign namespace
        insert_metadata(
            db_path,
            "/some/other/workdir/.snakemake",
            target="theirs.txt",
            rule="their_rule",
            starttime=2000.0,
            endtime=2010.0,
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())

        assert len(records) == 1
        assert records[0].rule == "our_rule"

    def test_iterate_metadata_code_hash_matches_fs(self, tmp_path: Path) -> None:
        """Code hash uses the same algorithm as the FS backend.

        The algorithm is: normalize whitespace, then SHA256 hex digest[:16].
        """
        db_path, namespace = create_test_db(tmp_path)
        code = "bwa  mem   {input}\n\t> {output}"
        insert_metadata(
            db_path,
            namespace,
            target="hash_test.txt",
            rule="hash_rule",
            starttime=1000.0,
            endtime=1010.0,
            code=code,
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())

        # Compute expected hash the same way the FS backend does
        normalized = " ".join(code.split())
        expected_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        assert len(records) == 1
        assert records[0].code_hash == expected_hash

    def test_iterate_metadata_with_input_json(self, tmp_path: Path) -> None:
        """Input JSON column is parsed without crashing even if files don't exist."""
        db_path, namespace = create_test_db(tmp_path)
        input_files = ["/nonexistent/file1.txt", "/nonexistent/file2.txt"]
        insert_metadata(
            db_path,
            namespace,
            target="input_test.txt",
            rule="input_rule",
            starttime=1000.0,
            endtime=1010.0,
            input_json=json.dumps(input_files),
        )
        backend = make_backend(tmp_path)
        records = list(backend.iterate_metadata())

        assert len(records) == 1
        # input_size should be None since files don't exist
        assert records[0].input_size is None

    def test_iterate_metadata_with_malformed_input_json(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed input JSON is logged and doesn't crash."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="bad_json.txt",
            rule="bad_input_rule",
            starttime=1000.0,
            endtime=1010.0,
            input_json="not valid json {{{",
        )
        backend = make_backend(tmp_path)
        with caplog.at_level(logging.DEBUG, logger="snakesee.persistence.db"):
            records = list(backend.iterate_metadata())

        assert len(records) == 1
        assert records[0].input_size is None
        assert any("input JSON" in msg for msg in caplog.messages)

    def test_iterate_metadata_with_progress_callback(self, tmp_path: Path) -> None:
        """Progress callback is invoked with current and total counts."""
        db_path, namespace = create_test_db(tmp_path)
        for i in range(3):
            insert_metadata(
                db_path,
                namespace,
                target=f"out_{i}.txt",
                rule=f"rule_{i}",
                starttime=1000.0 + i,
                endtime=1010.0 + i,
            )

        backend = make_backend(tmp_path)
        calls: list[tuple[int, int]] = []
        records = list(
            backend.iterate_metadata(progress_callback=lambda c, t: calls.append((c, t)))
        )

        assert len(records) == 3
        assert len(calls) == 3
        # All calls should have total=3
        assert all(t == 3 for _, t in calls)
        # Current should increment
        assert [c for c, _ in calls] == [1, 2, 3]


# =============================================================================
# Lock tests
# =============================================================================


class TestDbPersistenceLocks:
    """Tests for has_locks."""

    def test_has_locks_false_when_empty(self, tmp_path: Path) -> None:
        """No lock rows means has_locks returns False."""
        create_test_db(tmp_path)
        backend = make_backend(tmp_path)
        assert backend.has_locks() is False

    def test_has_locks_true_when_rows_exist(self, tmp_path: Path) -> None:
        """Lock rows for our namespace means has_locks returns True."""
        db_path, namespace = create_test_db(tmp_path)
        insert_lock(db_path, namespace, "output.txt", "output")
        backend = make_backend(tmp_path)
        assert backend.has_locks() is True

    def test_has_locks_ignores_foreign_namespace(self, tmp_path: Path) -> None:
        """Lock rows for a different namespace are ignored."""
        db_path, _namespace = create_test_db(tmp_path)
        insert_lock(db_path, "/other/.snakemake", "output.txt", "output")
        backend = make_backend(tmp_path)
        assert backend.has_locks() is False


# =============================================================================
# Incomplete job tests
# =============================================================================


class TestDbPersistenceIncomplete:
    """Tests for has_incomplete_jobs and iterate_incomplete_jobs."""

    def test_has_incomplete_false_when_none(self, tmp_path: Path) -> None:
        """No incomplete rows means has_incomplete_jobs returns False."""
        create_test_db(tmp_path)
        backend = make_backend(tmp_path)
        assert backend.has_incomplete_jobs() is False

    def test_has_incomplete_true_when_marked(self, tmp_path: Path) -> None:
        """An incomplete=1 row means has_incomplete_jobs returns True."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="running.txt",
            rule="slow_rule",
            starttime=1000.0,
            incomplete=True,
        )
        backend = make_backend(tmp_path)
        assert backend.has_incomplete_jobs() is True

    def test_iterate_incomplete_returns_jobs(self, tmp_path: Path) -> None:
        """Incomplete rows are returned with correct fields."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="running.txt",
            rule="slow_rule",
            starttime=1000.0,
            incomplete=True,
            external_jobid="cluster-123",
        )
        backend = make_backend(tmp_path)
        jobs = list(backend.iterate_incomplete_jobs())

        assert len(jobs) == 1
        job = jobs[0]
        assert job.rule == "slow_rule"
        assert job.start_time == 1000.0
        assert job.output_file == Path("running.txt")
        assert job.external_jobid == "cluster-123"

    def test_iterate_incomplete_filters_by_min_start_time(self, tmp_path: Path) -> None:
        """Only jobs at or after min_start_time are returned."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="old.txt",
            rule="old_rule",
            starttime=500.0,
            incomplete=True,
        )
        insert_metadata(
            db_path,
            namespace,
            target="new.txt",
            rule="new_rule",
            starttime=1500.0,
            incomplete=True,
        )
        insert_metadata(
            db_path,
            namespace,
            target="no_time.txt",
            rule="no_time_rule",
            incomplete=True,
        )
        backend = make_backend(tmp_path)
        jobs = list(backend.iterate_incomplete_jobs(min_start_time=1000.0))

        rules = {j.rule for j in jobs}
        assert "new_rule" in rules
        assert "no_time_rule" not in rules  # NULL starttime is excluded as stale
        assert "old_rule" not in rules

    def test_iterate_incomplete_ignores_foreign_namespace(self, tmp_path: Path) -> None:
        """Incomplete rows from a different namespace are ignored."""
        db_path, namespace = create_test_db(tmp_path)
        insert_metadata(
            db_path,
            namespace,
            target="ours.txt",
            rule="our_rule",
            starttime=1000.0,
            incomplete=True,
        )
        insert_metadata(
            db_path,
            "/other/.snakemake",
            target="theirs.txt",
            rule="their_rule",
            starttime=2000.0,
            incomplete=True,
        )
        backend = make_backend(tmp_path)
        jobs = list(backend.iterate_incomplete_jobs())

        assert len(jobs) == 1
        assert jobs[0].rule == "our_rule"


# =============================================================================
# Connection handling tests
# =============================================================================


class TestDbPersistenceSpecialPaths:
    """Tests for paths with special characters."""

    def test_connect_with_question_mark_in_path(self, tmp_path: Path) -> None:
        """Paths containing '?' must be percent-encoded for SQLite URI mode."""
        weird_dir = tmp_path / "workflow?test"
        smk_dir = weird_dir / ".snakemake"
        smk_dir.mkdir(parents=True)
        db_path = smk_dir / "metadata.db"
        namespace = str(smk_dir.resolve())

        conn = sqlite3.connect(str(db_path))
        create_metadata_schema(conn)
        conn.execute(
            """INSERT INTO snakemake_metadata
               (namespace, target, rule, starttime, endtime, incomplete,
                record_format_version)
               VALUES (?, ?, ?, ?, ?, 0, 6)""",
            (namespace, "out.txt", "align", 1000.0, 1060.0),
        )
        conn.commit()
        conn.close()

        clear_exists_cache()
        backend = DbPersistence(WorkflowPaths(weird_dir))
        records = list(backend.iterate_metadata())
        assert len(records) == 1
        assert records[0].rule == "align"

    def test_connect_with_hash_in_path(self, tmp_path: Path) -> None:
        """Paths containing '#' must be percent-encoded for SQLite URI mode."""
        weird_dir = tmp_path / "workflow#2"
        smk_dir = weird_dir / ".snakemake"
        smk_dir.mkdir(parents=True)
        db_path = smk_dir / "metadata.db"
        namespace = str(smk_dir.resolve())

        conn = sqlite3.connect(str(db_path))
        create_metadata_schema(conn)
        conn.execute(
            """INSERT INTO snakemake_metadata
               (namespace, target, rule, starttime, endtime, incomplete,
                record_format_version)
               VALUES (?, ?, ?, ?, ?, 0, 6)""",
            (namespace, "out.txt", "sort", 2000.0, 2030.0),
        )
        conn.commit()
        conn.close()

        clear_exists_cache()
        backend = DbPersistence(WorkflowPaths(weird_dir))
        records = list(backend.iterate_metadata())
        assert len(records) == 1
        assert records[0].rule == "sort"


class TestDbPersistenceConnection:
    """Tests for graceful handling of missing or corrupt DB files."""

    def test_missing_db_file(self, tmp_path: Path) -> None:
        """When the DB file doesn't exist, methods return empty/False gracefully."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir(parents=True)
        # Don't create metadata.db
        backend = make_backend(tmp_path)

        assert list(backend.iterate_metadata()) == []
        assert backend.has_locks() is False
        assert backend.has_incomplete_jobs() is False
        assert list(backend.iterate_incomplete_jobs()) == []

    def test_corrupt_db_file(self, tmp_path: Path) -> None:
        """When the DB file is corrupt, methods return empty/False gracefully."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir(parents=True)
        db_path = smk_dir / "metadata.db"
        db_path.write_text("this is not a sqlite database")

        backend = make_backend(tmp_path)

        assert list(backend.iterate_metadata()) == []
        assert backend.has_locks() is False
        assert backend.has_incomplete_jobs() is False
        assert list(backend.iterate_incomplete_jobs()) == []
