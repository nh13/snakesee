"""Tests for WorkflowPaths."""

from pathlib import Path

import pytest

from snakesee.state.paths import WorkflowPaths


class TestWorkflowPathsProperties:
    """Tests for WorkflowPaths computed properties."""

    def test_snakemake_dir(self, tmp_path: Path) -> None:
        """Test snakemake_dir property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.snakemake_dir == tmp_path / ".snakemake"

    def test_metadata_dir(self, tmp_path: Path) -> None:
        """Test metadata_dir property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.metadata_dir == tmp_path / ".snakemake" / "metadata"

    def test_log_dir(self, tmp_path: Path) -> None:
        """Test log_dir property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.log_dir == tmp_path / ".snakemake" / "log"

    def test_incomplete_dir(self, tmp_path: Path) -> None:
        """Test incomplete_dir property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.incomplete_dir == tmp_path / ".snakemake" / "incomplete"

    def test_locks_dir(self, tmp_path: Path) -> None:
        """Test locks_dir property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.locks_dir == tmp_path / ".snakemake" / "locks"

    def test_events_file(self, tmp_path: Path) -> None:
        """Test events_file property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.events_file == tmp_path / ".snakesee_events.jsonl"

    def test_validation_log(self, tmp_path: Path) -> None:
        """Test validation_log property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.validation_log == tmp_path / ".snakesee_validation.log"

    def test_default_profile(self, tmp_path: Path) -> None:
        """Test default_profile property."""
        paths = WorkflowPaths(tmp_path)
        assert paths.default_profile == tmp_path / ".snakesee-profile.json"


class TestWorkflowPathsExistence:
    """Tests for existence check properties."""

    def test_exists_false_when_no_snakemake_dir(self, tmp_path: Path) -> None:
        """Test exists is False when .snakemake doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.exists is False

    def test_exists_true_when_snakemake_dir_exists(self, tmp_path: Path) -> None:
        """Test exists is True when .snakemake exists."""
        (tmp_path / ".snakemake").mkdir()
        paths = WorkflowPaths(tmp_path)
        assert paths.exists is True

    def test_has_metadata_false_when_no_metadata(self, tmp_path: Path) -> None:
        """Test has_metadata is False when metadata dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.has_metadata is False

    def test_has_metadata_true_when_exists(self, tmp_path: Path) -> None:
        """Test has_metadata is True when metadata dir exists."""
        (tmp_path / ".snakemake" / "metadata").mkdir(parents=True)
        paths = WorkflowPaths(tmp_path)
        assert paths.has_metadata is True

    def test_has_logs_false_when_no_log_dir(self, tmp_path: Path) -> None:
        """Test has_logs is False when log dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.has_logs is False

    def test_has_logs_false_when_empty(self, tmp_path: Path) -> None:
        """Test has_logs is False when log dir is empty."""
        (tmp_path / ".snakemake" / "log").mkdir(parents=True)
        paths = WorkflowPaths(tmp_path)
        assert paths.has_logs is False

    def test_has_logs_true_when_logs_exist(self, tmp_path: Path) -> None:
        """Test has_logs is True when logs exist."""
        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        (log_dir / "test.snakemake.log").touch()
        paths = WorkflowPaths(tmp_path)
        assert paths.has_logs is True

    def test_has_events_false_when_no_file(self, tmp_path: Path) -> None:
        """Test has_events is False when events file doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.has_events is False

    def test_has_events_false_when_empty(self, tmp_path: Path) -> None:
        """Test has_events is False when events file is empty."""
        (tmp_path / ".snakesee_events.jsonl").touch()
        paths = WorkflowPaths(tmp_path)
        assert paths.has_events is False

    def test_has_events_true_when_has_content(self, tmp_path: Path) -> None:
        """Test has_events is True when events file has content."""
        (tmp_path / ".snakesee_events.jsonl").write_text('{"event": "test"}\n')
        paths = WorkflowPaths(tmp_path)
        assert paths.has_events is True


class TestLogDiscovery:
    """Tests for log file discovery methods."""

    def test_find_latest_log_returns_none_when_no_logs(self, tmp_path: Path) -> None:
        """Test find_latest_log returns None when no logs exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.find_latest_log() is None

    def test_find_latest_log_returns_none_when_empty_dir(self, tmp_path: Path) -> None:
        """Test find_latest_log returns None when log dir is empty."""
        (tmp_path / ".snakemake" / "log").mkdir(parents=True)
        paths = WorkflowPaths(tmp_path)
        assert paths.find_latest_log() is None

    def test_find_latest_log_returns_newest(self, tmp_path: Path) -> None:
        """Test find_latest_log returns the most recently modified log."""
        import time

        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)

        old_log = log_dir / "old.snakemake.log"
        old_log.touch()

        # Small delay to ensure different mtime
        time.sleep(0.01)

        new_log = log_dir / "new.snakemake.log"
        new_log.touch()

        paths = WorkflowPaths(tmp_path)
        assert paths.find_latest_log() == new_log

    def test_find_all_logs_returns_empty_when_no_logs(self, tmp_path: Path) -> None:
        """Test find_all_logs returns empty list when no logs exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.find_all_logs() == []

    def test_find_all_logs_returns_sorted(self, tmp_path: Path) -> None:
        """Test find_all_logs returns logs sorted by mtime."""
        import time

        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)

        log1 = log_dir / "first.snakemake.log"
        log1.touch()
        time.sleep(0.01)

        log2 = log_dir / "second.snakemake.log"
        log2.touch()

        paths = WorkflowPaths(tmp_path)
        logs = paths.find_all_logs()
        assert logs == [log1, log2]

    def test_find_logs_sorted_newest_first(self, tmp_path: Path) -> None:
        """Test find_logs_sorted_newest_first returns logs in reverse order."""
        import time

        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)

        log1 = log_dir / "first.snakemake.log"
        log1.touch()
        time.sleep(0.01)

        log2 = log_dir / "second.snakemake.log"
        log2.touch()

        paths = WorkflowPaths(tmp_path)
        logs = paths.find_logs_sorted_newest_first()
        assert logs == [log2, log1]


class TestMetadataDiscovery:
    """Tests for metadata file discovery."""

    def test_get_metadata_files_empty_when_no_dir(self, tmp_path: Path) -> None:
        """Test get_metadata_files yields nothing when dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert list(paths.get_metadata_files()) == []

    def test_get_metadata_files_yields_files(self, tmp_path: Path) -> None:
        """Test get_metadata_files yields all files."""
        metadata_dir = tmp_path / ".snakemake" / "metadata"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "file1.json").touch()
        (metadata_dir / "file2.json").touch()

        paths = WorkflowPaths(tmp_path)
        files = list(paths.get_metadata_files())
        assert len(files) == 2

    def test_count_metadata_files(self, tmp_path: Path) -> None:
        """Test count_metadata_files returns correct count."""
        metadata_dir = tmp_path / ".snakemake" / "metadata"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "file1.json").touch()
        (metadata_dir / "file2.json").touch()
        (metadata_dir / "file3.json").touch()

        paths = WorkflowPaths(tmp_path)
        assert paths.count_metadata_files() == 3

    def test_count_metadata_files_zero_when_no_dir(self, tmp_path: Path) -> None:
        """Test count_metadata_files returns 0 when dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.count_metadata_files() == 0


class TestValidation:
    """Tests for validation method."""

    def test_validate_raises_when_no_snakemake_dir(self, tmp_path: Path) -> None:
        """Test validate raises ValueError when .snakemake doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        with pytest.raises(ValueError, match="No .snakemake directory"):
            paths.validate()

    def test_validate_succeeds_when_snakemake_exists(self, tmp_path: Path) -> None:
        """Test validate succeeds when .snakemake exists."""
        (tmp_path / ".snakemake").mkdir()
        paths = WorkflowPaths(tmp_path)
        paths.validate()  # Should not raise


class TestFrozen:
    """Tests that WorkflowPaths is frozen."""

    def test_frozen(self, tmp_path: Path) -> None:
        """Test that WorkflowPaths is frozen."""
        paths = WorkflowPaths(tmp_path)
        with pytest.raises(AttributeError):
            paths.workflow_dir = tmp_path / "other"  # type: ignore[misc]
