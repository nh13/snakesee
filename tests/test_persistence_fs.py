"""Tests for filesystem persistence backend."""

import base64
import json
import os
import time
from pathlib import Path

import pytest

from snakesee.persistence.fs import FsPersistence
from snakesee.state.paths import WorkflowPaths


class TestFsPersistenceMetadata:
    """Tests for metadata iteration."""

    def test_iterate_empty_metadata(self, tmp_path: Path) -> None:
        """Test iterating when metadata dir is empty."""
        smk_dir = tmp_path / ".snakemake" / "metadata"
        smk_dir.mkdir(parents=True)
        backend = FsPersistence(WorkflowPaths(tmp_path))
        records = list(backend.iterate_metadata())
        assert records == []

    def test_iterate_metadata_returns_records(self, tmp_path: Path) -> None:
        """Test that metadata files are parsed into MetadataRecord."""
        meta_dir = tmp_path / ".snakemake" / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / "abc123").write_text(
            json.dumps(
                {
                    "rule": "align",
                    "starttime": 1000.0,
                    "endtime": 1100.0,
                    "code": "bwa mem ref.fa in.fq > out.bam",
                }
            )
        )

        backend = FsPersistence(WorkflowPaths(tmp_path))
        records = list(backend.iterate_metadata())
        assert len(records) == 1
        assert records[0].rule == "align"
        assert records[0].duration == pytest.approx(100.0)
        assert records[0].code_hash is not None

    def test_iterate_metadata_skips_missing_rule(self, tmp_path: Path) -> None:
        """Test that metadata files without rule are skipped."""
        meta_dir = tmp_path / ".snakemake" / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / "no_rule").write_text(
            json.dumps(
                {
                    "starttime": 1000.0,
                    "endtime": 1100.0,
                }
            )
        )

        backend = FsPersistence(WorkflowPaths(tmp_path))
        records = list(backend.iterate_metadata())
        assert records == []

    def test_iterate_metadata_with_wildcards(self, tmp_path: Path) -> None:
        """Test that wildcards are extracted from metadata."""
        meta_dir = tmp_path / ".snakemake" / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / "with_wc").write_text(
            json.dumps(
                {
                    "rule": "align",
                    "starttime": 1000.0,
                    "endtime": 1100.0,
                    "wildcards": {"sample": "A", "lane": "1"},
                }
            )
        )

        backend = FsPersistence(WorkflowPaths(tmp_path))
        records = list(backend.iterate_metadata())
        assert records[0].wildcards == {"sample": "A", "lane": "1"}


class TestFsPersistenceLocks:
    """Tests for lock detection."""

    def test_has_locks_false_when_dir_missing(self, tmp_path: Path) -> None:
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_locks() is False

    def test_has_locks_false_when_dir_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".snakemake" / "locks").mkdir(parents=True)
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_locks() is False

    def test_has_locks_true_when_lock_files_exist(self, tmp_path: Path) -> None:
        locks_dir = tmp_path / ".snakemake" / "locks"
        locks_dir.mkdir(parents=True)
        (locks_dir / "some_lock").touch()
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_locks() is True


class TestFsPersistenceIncomplete:
    """Tests for incomplete job detection."""

    def test_has_incomplete_false_when_dir_missing(self, tmp_path: Path) -> None:
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_incomplete_jobs() is False

    def test_has_incomplete_false_when_dir_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".snakemake" / "incomplete").mkdir(parents=True)
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_incomplete_jobs() is False

    def test_has_incomplete_true_when_markers_exist(self, tmp_path: Path) -> None:
        inc_dir = tmp_path / ".snakemake" / "incomplete"
        inc_dir.mkdir(parents=True)
        encoded = base64.b64encode(b"output.bam").decode()
        (inc_dir / encoded).touch()
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_incomplete_jobs() is True

    def test_has_incomplete_false_when_only_migration_marker(self, tmp_path: Path) -> None:
        """migration_underway marker alone should not count as incomplete."""
        inc_dir = tmp_path / ".snakemake" / "incomplete"
        inc_dir.mkdir(parents=True)
        (inc_dir / "migration_underway").touch()
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_incomplete_jobs() is False

    def test_has_incomplete_false_when_only_subdirectories(self, tmp_path: Path) -> None:
        """Plain subdirectories should not count as incomplete jobs."""
        inc_dir = tmp_path / ".snakemake" / "incomplete"
        inc_dir.mkdir(parents=True)
        (inc_dir / "some_subdir").mkdir()
        backend = FsPersistence(WorkflowPaths(tmp_path))
        assert backend.has_incomplete_jobs() is False

    def test_iterate_incomplete_returns_jobs(self, tmp_path: Path) -> None:
        inc_dir = tmp_path / ".snakemake" / "incomplete"
        inc_dir.mkdir(parents=True)
        encoded = base64.b64encode(b"output.bam").decode()
        marker = inc_dir / encoded
        marker.touch()

        backend = FsPersistence(WorkflowPaths(tmp_path))
        jobs = list(backend.iterate_incomplete_jobs())
        assert len(jobs) == 1
        assert jobs[0].output_file == Path("output.bam")
        assert jobs[0].start_time is not None
        assert jobs[0].rule is None

    def test_iterate_incomplete_filters_by_min_start_time(self, tmp_path: Path) -> None:
        inc_dir = tmp_path / ".snakemake" / "incomplete"
        inc_dir.mkdir(parents=True)

        encoded = base64.b64encode(b"old.bam").decode()
        marker = inc_dir / encoded
        marker.touch()
        old_time = time.time() - 3600
        os.utime(marker, (old_time, old_time))

        backend = FsPersistence(WorkflowPaths(tmp_path))
        jobs = list(backend.iterate_incomplete_jobs(min_start_time=time.time() - 60))
        assert len(jobs) == 0
