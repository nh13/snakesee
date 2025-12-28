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


class TestLocksAndIncomplete:
    """Tests for locks and incomplete marker properties."""

    def test_has_locks_false_when_no_dir(self, tmp_path: Path) -> None:
        """Test has_locks is False when locks dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.has_locks is False

    def test_has_locks_false_when_empty(self, tmp_path: Path) -> None:
        """Test has_locks is False when locks dir is empty."""
        (tmp_path / ".snakemake" / "locks").mkdir(parents=True)
        paths = WorkflowPaths(tmp_path)
        assert paths.has_locks is False

    def test_has_locks_true_when_files_exist(self, tmp_path: Path) -> None:
        """Test has_locks is True when lock files exist."""
        locks_dir = tmp_path / ".snakemake" / "locks"
        locks_dir.mkdir(parents=True)
        (locks_dir / "0.input.lock").touch()
        paths = WorkflowPaths(tmp_path)
        assert paths.has_locks is True

    def test_has_incomplete_false_when_no_dir(self, tmp_path: Path) -> None:
        """Test has_incomplete is False when incomplete dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert paths.has_incomplete is False

    def test_has_incomplete_false_when_empty(self, tmp_path: Path) -> None:
        """Test has_incomplete is False when incomplete dir is empty."""
        (tmp_path / ".snakemake" / "incomplete").mkdir(parents=True)
        paths = WorkflowPaths(tmp_path)
        assert paths.has_incomplete is False

    def test_has_incomplete_true_when_markers_exist(self, tmp_path: Path) -> None:
        """Test has_incomplete is True when incomplete markers exist."""
        incomplete_dir = tmp_path / ".snakemake" / "incomplete"
        incomplete_dir.mkdir(parents=True)
        (incomplete_dir / "c29tZV9vdXRwdXRfZmlsZQ==").touch()  # base64 encoded
        paths = WorkflowPaths(tmp_path)
        assert paths.has_incomplete is True

    def test_get_incomplete_markers_empty_when_no_dir(self, tmp_path: Path) -> None:
        """Test get_incomplete_markers yields nothing when dir doesn't exist."""
        paths = WorkflowPaths(tmp_path)
        assert list(paths.get_incomplete_markers()) == []

    def test_get_incomplete_markers_skips_migration_underway(self, tmp_path: Path) -> None:
        """Test get_incomplete_markers skips migration_underway file."""
        incomplete_dir = tmp_path / ".snakemake" / "incomplete"
        incomplete_dir.mkdir(parents=True)
        (incomplete_dir / "migration_underway").touch()
        (incomplete_dir / "c29tZV9vdXRwdXRfZmlsZQ==").touch()
        paths = WorkflowPaths(tmp_path)
        markers = list(paths.get_incomplete_markers())
        assert len(markers) == 1
        assert markers[0].name != "migration_underway"

    def test_decode_incomplete_marker_valid(self, tmp_path: Path) -> None:
        """Test decoding valid base64 marker."""
        import base64

        output_path = "/path/to/output.txt"
        encoded = base64.b64encode(output_path.encode()).decode()
        marker = tmp_path / encoded
        marker.touch()
        paths = WorkflowPaths(tmp_path)
        decoded = paths.decode_incomplete_marker(marker)
        assert decoded == Path(output_path)

    def test_decode_incomplete_marker_invalid(self, tmp_path: Path) -> None:
        """Test decoding invalid base64 marker returns None."""
        marker = tmp_path / "not_valid_base64!!!"
        marker.touch()
        paths = WorkflowPaths(tmp_path)
        decoded = paths.decode_incomplete_marker(marker)
        assert decoded is None


class TestJobLogDiscovery:
    """Tests for job log discovery methods."""

    def test_get_job_log_returns_none_when_no_logs(self, tmp_path: Path) -> None:
        """Test get_job_log returns None when no matching logs exist."""
        paths = WorkflowPaths(tmp_path)
        result = paths.get_job_log("align")
        assert result is None

    def test_get_job_log_finds_log_in_snakemake_log_dir(self, tmp_path: Path) -> None:
        """Test get_job_log finds logs in .snakemake/log/."""
        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "align_sample_A.log"
        log_file.write_text("log content")
        paths = WorkflowPaths(tmp_path)
        result = paths.get_job_log("align")
        assert result == log_file

    def test_get_job_log_finds_log_by_job_id(self, tmp_path: Path) -> None:
        """Test get_job_log finds logs by job ID."""
        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "job42.log"
        log_file.write_text("log content")
        paths = WorkflowPaths(tmp_path)
        result = paths.get_job_log("some_rule", job_id=42)
        assert result == log_file

    def test_get_job_log_finds_log_in_logs_dir(self, tmp_path: Path) -> None:
        """Test get_job_log finds logs in logs/ directory."""
        log_dir = tmp_path / "logs" / "align"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "sample_A.log"
        log_file.write_text("log content")
        # Also create .snakemake for valid workflow
        (tmp_path / ".snakemake").mkdir()
        paths = WorkflowPaths(tmp_path)
        result = paths.get_job_log("align")
        assert result == log_file

    def test_get_job_log_finds_log_by_wildcard(self, tmp_path: Path) -> None:
        """Test get_job_log finds logs by wildcard value."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "sample_FASTQ_A.log"
        log_file.write_text("log content")
        (tmp_path / ".snakemake").mkdir()
        paths = WorkflowPaths(tmp_path)
        result = paths.get_job_log("align", wildcards={"sample": "FASTQ_A"})
        assert result == log_file

    def test_get_job_log_returns_newest(self, tmp_path: Path) -> None:
        """Test get_job_log returns newest log when multiple match."""
        import time

        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        old_log = log_dir / "align_1.log"
        old_log.write_text("old")
        time.sleep(0.01)
        new_log = log_dir / "align_2.log"
        new_log.write_text("new")
        paths = WorkflowPaths(tmp_path)
        result = paths.get_job_log("align")
        assert result == new_log


class TestProfileDiscovery:
    """Tests for profile discovery."""

    def test_find_profile_returns_none_when_no_profile(self, tmp_path: Path) -> None:
        """Test find_profile returns None when no profile exists."""
        paths = WorkflowPaths(tmp_path)
        result = paths.find_profile()
        assert result is None

    def test_find_profile_finds_in_workflow_dir(self, tmp_path: Path) -> None:
        """Test find_profile finds profile in workflow directory."""
        profile = tmp_path / ".snakesee-profile.json"
        profile.write_text('{"rules": {}}')
        paths = WorkflowPaths(tmp_path)
        result = paths.find_profile()
        assert result == profile

    def test_find_profile_searches_parent_dirs(self, tmp_path: Path) -> None:
        """Test find_profile searches parent directories."""
        # Create profile in parent
        profile = tmp_path / ".snakesee-profile.json"
        profile.write_text('{"rules": {}}')
        # Create subdirectory workflow
        subdir = tmp_path / "workflows" / "my_workflow"
        subdir.mkdir(parents=True)
        paths = WorkflowPaths(subdir)
        result = paths.find_profile()
        assert result == profile

    def test_find_profile_respects_max_levels(self, tmp_path: Path) -> None:
        """Test find_profile respects max_levels parameter."""
        # Create deeply nested workflow
        deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g"
        deep_dir.mkdir(parents=True)
        # Profile at top level
        profile = tmp_path / ".snakesee-profile.json"
        profile.write_text('{"rules": {}}')
        paths = WorkflowPaths(deep_dir)
        # With max_levels=3, should not find profile 7 levels up
        result = paths.find_profile(max_levels=3)
        assert result is None
        # With max_levels=10, should find it
        result = paths.find_profile(max_levels=10)
        assert result == profile


class TestExistsCache:
    """Tests for the exists cache functions."""

    def test_clear_exists_cache(self, tmp_path: Path) -> None:
        """Test clear_exists_cache clears the cache."""
        from snakesee.state.paths import _cached_exists
        from snakesee.state.paths import clear_exists_cache

        # Populate cache
        _cached_exists(tmp_path)
        # Clear it
        clear_exists_cache()
        # Cache should be cleared (tested indirectly - function should not raise)

    def test_cached_exists_caches_result(self, tmp_path: Path) -> None:
        """Test _cached_exists caches results."""
        from snakesee.state.paths import _cached_exists
        from snakesee.state.paths import _exists_cache
        from snakesee.state.paths import clear_exists_cache

        clear_exists_cache()
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # First call should cache
        result1 = _cached_exists(test_dir)
        assert result1 is True
        assert str(test_dir) in _exists_cache

        # Even if we delete the dir, cached result is returned
        test_dir.rmdir()
        result2 = _cached_exists(test_dir, ttl=10.0)  # Long TTL
        assert result2 is True  # Still cached as True

        # Clear cache and check again
        clear_exists_cache()
        result3 = _cached_exists(test_dir)
        assert result3 is False  # Now correctly False
