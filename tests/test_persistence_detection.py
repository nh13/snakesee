"""Tests for persistence backend auto-detection."""

import json
import sqlite3
from pathlib import Path

from snakesee.persistence.backend import detect_backend
from snakesee.persistence.db import DbPersistence
from snakesee.persistence.fs import FsPersistence


class TestDetectBackend:
    """Tests for detect_backend factory."""

    def test_returns_fs_when_no_db(self, tmp_path: Path) -> None:
        """Test filesystem backend when no metadata.db exists."""
        smk_dir = tmp_path / ".snakemake" / "metadata"
        smk_dir.mkdir(parents=True)

        backend = detect_backend(tmp_path)
        assert isinstance(backend, FsPersistence)

    def test_returns_db_when_db_exists(self, tmp_path: Path) -> None:
        """Test SQLite backend when metadata.db exists."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir(parents=True)
        db_path = smk_dir / "metadata.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        backend = detect_backend(tmp_path)
        assert isinstance(backend, DbPersistence)

    def test_prefers_db_when_both_exist(self, tmp_path: Path) -> None:
        """Test DB is preferred when both backends are available."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir(parents=True)

        # Create filesystem metadata
        meta_dir = smk_dir / "metadata"
        meta_dir.mkdir()
        (meta_dir / "abc123").write_text(
            json.dumps(
                {
                    "rule": "align",
                    "starttime": 1000.0,
                    "endtime": 1100.0,
                }
            )
        )

        # Create DB
        db_path = smk_dir / "metadata.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        backend = detect_backend(tmp_path)
        assert isinstance(backend, DbPersistence)

    def test_returns_fs_when_no_snakemake_dir(self, tmp_path: Path) -> None:
        """Test fallback to FS when .snakemake/ doesn't exist at all."""
        backend = detect_backend(tmp_path)
        assert isinstance(backend, FsPersistence)

    def test_detects_db_created_after_cache(self, tmp_path: Path) -> None:
        """Test that detect_backend sees a DB created after the cache is populated.

        WorkflowPaths caches exists() results, but detect_backend should use
        a direct filesystem check so it picks up a newly-created metadata.db.
        """
        from snakesee.state.paths import WorkflowPaths
        from snakesee.state.paths import clear_exists_cache

        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir(parents=True)

        # Populate the cache — no DB yet
        clear_exists_cache()
        paths = WorkflowPaths(tmp_path)
        assert not paths.has_metadata_db  # cache: no DB

        # Now create the DB file after the cache miss
        db_path = smk_dir / "metadata.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        # detect_backend should find the DB despite the stale cache
        backend = detect_backend(tmp_path)
        assert isinstance(backend, DbPersistence)
