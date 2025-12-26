"""Tests for monitor parser functions."""

import json
from pathlib import Path

from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from snakesee.models import WorkflowStatus
from snakesee.parser import IncrementalLogReader
from snakesee.parser import _parse_wildcards
from snakesee.parser import collect_rule_timing_stats
from snakesee.parser import collect_wildcard_timing_stats
from snakesee.parser import is_workflow_running
from snakesee.parser import parse_failed_jobs_from_log
from snakesee.parser import parse_metadata_files
from snakesee.parser import parse_progress_from_log
from snakesee.parser import parse_running_jobs_from_log
from snakesee.parser import parse_workflow_state
from snakesee.state.paths import WorkflowPaths


class TestParseWildcards:
    """Tests for _parse_wildcards function."""

    def test_single_wildcard(self) -> None:
        """Test parsing a single wildcard."""
        result = _parse_wildcards("sample=A")
        assert result == {"sample": "A"}

    def test_multiple_wildcards(self) -> None:
        """Test parsing multiple wildcards."""
        result = _parse_wildcards("sample=A, batch=1")
        assert result == {"sample": "A", "batch": "1"}

    def test_wildcards_with_spaces(self) -> None:
        """Test parsing wildcards with extra spaces."""
        result = _parse_wildcards("sample = A , batch = 1")
        assert result == {"sample": "A", "batch": "1"}

    def test_wildcard_with_path(self) -> None:
        """Test parsing wildcard with path-like value."""
        result = _parse_wildcards("sample=data/samples/A.fastq")
        assert result == {"sample": "data/samples/A.fastq"}

    def test_empty_string(self) -> None:
        """Test parsing empty string."""
        result = _parse_wildcards("")
        assert result == {}


class TestFindLatestLog:
    """Tests for WorkflowPaths.find_latest_log function."""

    def test_no_log_dir(self, tmp_path: Path) -> None:
        """Test when log directory doesn't exist."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir()
        paths = WorkflowPaths(tmp_path)
        assert paths.find_latest_log() is None

    def test_empty_log_dir(self, snakemake_dir: Path) -> None:
        """Test when log directory is empty."""
        paths = WorkflowPaths(snakemake_dir.parent)
        assert paths.find_latest_log() is None

    def test_single_log(self, snakemake_dir: Path) -> None:
        """Test with a single log file."""
        log_file = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        log_file.write_text("test")
        paths = WorkflowPaths(snakemake_dir.parent)
        assert paths.find_latest_log() == log_file

    def test_multiple_logs(self, snakemake_dir: Path) -> None:
        """Test returns most recent log."""
        import time

        old_log = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        old_log.write_text("old")
        time.sleep(0.1)

        new_log = snakemake_dir / "log" / "2024-01-02T120000.000000.snakemake.log"
        new_log.write_text("new")

        paths = WorkflowPaths(snakemake_dir.parent)
        assert paths.find_latest_log() == new_log


class TestParseProgressFromLog:
    """Tests for parse_progress_from_log function."""

    def test_no_progress_lines(self, tmp_path: Path) -> None:
        """Test when log has no progress lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Building DAG of jobs...\nUsing shell: /bin/bash")
        completed, total = parse_progress_from_log(log_file)
        assert completed == 0
        assert total == 0

    def test_single_progress_line(self, tmp_path: Path) -> None:
        """Test parsing a single progress line."""
        log_file = tmp_path / "test.log"
        log_file.write_text("5 of 10 steps (50%) done")
        completed, total = parse_progress_from_log(log_file)
        assert completed == 5
        assert total == 10

    def test_multiple_progress_lines(self, tmp_path: Path) -> None:
        """Test returns the latest progress."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "1 of 10 steps (10%) done\n"
            "rule align:\n"
            "5 of 10 steps (50%) done\n"
            "rule merge:\n"
            "8 of 10 steps (80%) done"
        )
        completed, total = parse_progress_from_log(log_file)
        assert completed == 8
        assert total == 10

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test with nonexistent file."""
        log_file = tmp_path / "nonexistent.log"
        completed, total = parse_progress_from_log(log_file)
        assert completed == 0
        assert total == 0


class TestParseMetadataFiles:
    """Tests for parse_metadata_files function."""

    def test_empty_dir(self, snakemake_dir: Path) -> None:
        """Test with empty metadata directory."""
        jobs = list(parse_metadata_files(snakemake_dir / "metadata"))
        assert jobs == []

    def test_valid_metadata(self, snakemake_dir: Path) -> None:
        """Test parsing valid metadata."""
        metadata = {
            "rule": "align",
            "starttime": 1000.0,
            "endtime": 1100.0,
        }
        meta_file = snakemake_dir / "metadata" / "dGVzdA=="  # base64 encoded
        meta_file.write_text(json.dumps(metadata))

        jobs = list(parse_metadata_files(snakemake_dir / "metadata"))
        assert len(jobs) == 1
        assert jobs[0].rule == "align"
        assert jobs[0].start_time == 1000.0
        assert jobs[0].end_time == 1100.0

    def test_incomplete_metadata(self, snakemake_dir: Path) -> None:
        """Test skips metadata without required fields."""
        metadata = {"rule": "align"}  # Missing times
        meta_file = snakemake_dir / "metadata" / "incomplete"
        meta_file.write_text(json.dumps(metadata))

        jobs = list(parse_metadata_files(snakemake_dir / "metadata"))
        assert jobs == []

    def test_invalid_json(self, snakemake_dir: Path) -> None:
        """Test skips invalid JSON files."""
        meta_file = snakemake_dir / "metadata" / "invalid"
        meta_file.write_text("not json")

        jobs = list(parse_metadata_files(snakemake_dir / "metadata"))
        assert jobs == []


class TestIsWorkflowRunning:
    """Tests for is_workflow_running function."""

    def test_no_locks_dir(self, tmp_path: Path) -> None:
        """Test when locks directory doesn't exist."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir()
        assert is_workflow_running(smk_dir) is False

    def test_empty_locks_dir(self, snakemake_dir: Path) -> None:
        """Test when locks directory is empty."""
        assert is_workflow_running(snakemake_dir) is False

    def test_with_lock_file(self, snakemake_dir: Path) -> None:
        """Test when lock file exists but no log (early startup)."""
        lock_file = snakemake_dir / "locks" / "0.input.lock"
        lock_file.write_text("/some/file\n")
        # No log file - should assume running (early startup)
        assert is_workflow_running(snakemake_dir) is True

    def test_with_lock_and_recent_log(self, snakemake_dir: Path) -> None:
        """Test when lock and recent log exist."""
        lock_file = snakemake_dir / "locks" / "0.input.lock"
        lock_file.write_text("/some/file\n")
        log_file = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        log_file.write_text("test")
        # Recent log - should be running
        assert is_workflow_running(snakemake_dir) is True

    def test_with_lock_and_stale_log(self, snakemake_dir: Path) -> None:
        """Test when lock exists but log is stale and no incomplete markers."""
        import os
        import time

        lock_file = snakemake_dir / "locks" / "0.input.lock"
        lock_file.write_text("/some/file\n")
        log_file = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        log_file.write_text("test")

        # Make log file appear old by setting mtime to 31 minutes ago
        old_time = time.time() - 1860
        os.utime(log_file, (old_time, old_time))

        # With default 1800s threshold and no incomplete markers, workflow should appear dead
        assert is_workflow_running(snakemake_dir) is False

        # With 2400s threshold, should still appear running
        assert is_workflow_running(snakemake_dir, stale_threshold=2400.0) is True

    def test_with_lock_stale_log_but_incomplete_markers(self, snakemake_dir: Path) -> None:
        """Test when lock exists, log is stale, but incomplete markers exist."""
        import os
        import time

        lock_file = snakemake_dir / "locks" / "0.input.lock"
        lock_file.write_text("/some/file\n")
        log_file = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        log_file.write_text("test")

        # Make log file appear old by setting mtime to 2 hours ago
        old_time = time.time() - 7200
        os.utime(log_file, (old_time, old_time))

        # Create incomplete marker (job in progress)
        incomplete_dir = snakemake_dir / "incomplete"
        incomplete_dir.mkdir(exist_ok=True)
        incomplete_marker = incomplete_dir / "c29tZV9vdXRwdXRfZmlsZQ=="  # base64 encoded
        incomplete_marker.write_text("")

        # When locks AND incomplete markers both exist, the workflow is running.
        # Incomplete markers are created when jobs START and removed when they FINISH,
        # so their presence combined with locks strongly indicates active execution.
        # (Log freshness is only used as a fallback when no incomplete markers exist)
        assert is_workflow_running(snakemake_dir) is True


class TestCollectRuleTimingStats:
    """Tests for collect_rule_timing_stats function."""

    def test_empty_dir(self, snakemake_dir: Path) -> None:
        """Test with empty metadata directory."""
        stats = collect_rule_timing_stats(snakemake_dir / "metadata")
        assert stats == {}

    def test_single_rule(self, snakemake_dir: Path) -> None:
        """Test collecting stats for a single rule."""
        for i in range(3):
            metadata = {
                "rule": "align",
                "starttime": 1000.0 + i * 200,
                "endtime": 1100.0 + i * 200,
            }
            meta_file = snakemake_dir / "metadata" / f"job{i}"
            meta_file.write_text(json.dumps(metadata))

        stats = collect_rule_timing_stats(snakemake_dir / "metadata")
        assert "align" in stats
        assert stats["align"].count == 3
        assert stats["align"].mean_duration == 100.0

    def test_multiple_rules(self, snakemake_dir: Path) -> None:
        """Test collecting stats for multiple rules."""
        metadata1 = {"rule": "align", "starttime": 1000.0, "endtime": 1100.0}
        metadata2 = {"rule": "sort", "starttime": 1100.0, "endtime": 1150.0}

        (snakemake_dir / "metadata" / "job1").write_text(json.dumps(metadata1))
        (snakemake_dir / "metadata" / "job2").write_text(json.dumps(metadata2))

        stats = collect_rule_timing_stats(snakemake_dir / "metadata")
        assert len(stats) == 2
        assert stats["align"].mean_duration == 100.0
        assert stats["sort"].mean_duration == 50.0


class TestParseRunningJobsFromLog:
    """Tests for parse_running_jobs_from_log function."""

    def test_no_running_jobs(self, snakemake_dir: Path) -> None:
        """Test when all jobs are finished."""
        log_content = """
rule align:
    jobid: 1
Finished job 1.
rule sort:
    jobid: 2
Finished job 2.
2 of 2 steps (100%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        running = parse_running_jobs_from_log(log_file)
        assert len(running) == 0

    def test_one_running_job(self, snakemake_dir: Path) -> None:
        """Test detecting a single running job."""
        log_content = """
rule align:
    jobid: 1
Finished job 1.
rule sort:
    jobid: 2
1 of 2 steps (50%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        running = parse_running_jobs_from_log(log_file)
        assert len(running) == 1
        assert running[0].rule == "sort"
        assert running[0].job_id == "2"

    def test_multiple_running_jobs(self, snakemake_dir: Path) -> None:
        """Test detecting multiple running jobs."""
        log_content = """
rule download:
    jobid: 1
Finished job 1.
rule align:
    jobid: 2
rule align:
    jobid: 3
rule sort:
    jobid: 4
1 of 4 steps (25%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        running = parse_running_jobs_from_log(log_file)
        assert len(running) == 3
        rules = {j.rule for j in running}
        assert rules == {"align", "sort"}
        job_ids = {j.job_id for j in running}
        assert job_ids == {"2", "3", "4"}

    def test_localrule(self, snakemake_dir: Path) -> None:
        """Test parsing localrule entries."""
        log_content = """
localrule all:
    jobid: 0
rule align:
    jobid: 1
0 of 2 steps (0%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        running = parse_running_jobs_from_log(log_file)
        assert len(running) == 2
        rules = {j.rule for j in running}
        assert "all" in rules
        assert "align" in rules

    def test_running_job_with_wildcards(self, snakemake_dir: Path) -> None:
        """Test detecting running job with wildcards."""
        log_content = """
rule align:
    input: data/sample_A.fastq
    output: results/aligned_A.bam
    wildcards: sample=A
    jobid: 1
0 of 1 steps (0%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        running = parse_running_jobs_from_log(log_file)
        assert len(running) == 1
        assert running[0].rule == "align"
        assert running[0].wildcards == {"sample": "A"}

    def test_running_job_with_multiple_wildcards(self, snakemake_dir: Path) -> None:
        """Test detecting running job with multiple wildcards."""
        log_content = """
rule align:
    wildcards: sample=A, batch=1
    jobid: 1
0 of 1 steps (0%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        running = parse_running_jobs_from_log(log_file)
        assert len(running) == 1
        assert running[0].wildcards == {"sample": "A", "batch": "1"}


class TestParseFailedJobsFromLog:
    """Tests for parse_failed_jobs_from_log function."""

    def test_no_failed_jobs(self, snakemake_dir: Path) -> None:
        """Test when no jobs have failed."""
        log_content = """
rule align:
    jobid: 1
Finished job 1.
rule sort:
    jobid: 2
Finished job 2.
2 of 2 steps (100%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        failed = parse_failed_jobs_from_log(log_file)
        assert len(failed) == 0

    def test_one_failed_job(self, snakemake_dir: Path) -> None:
        """Test detecting a single failed job."""
        log_content = """
rule align:
    jobid: 1
Error in rule align:
    Some error message
Shutting down, error in workflow
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        failed = parse_failed_jobs_from_log(log_file)
        assert len(failed) == 1
        assert failed[0].rule == "align"
        assert failed[0].job_id == "1"

    def test_multiple_failed_jobs_keep_going(self, snakemake_dir: Path) -> None:
        """Test detecting multiple failed jobs (--keep-going mode)."""
        log_content = """
rule align:
    jobid: 1
Finished job 1.
rule sort:
    jobid: 2
Error in rule sort:
    Sort failed
rule merge:
    jobid: 3
Error in rule merge:
    Merge failed
1 of 3 steps (33%) done
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        failed = parse_failed_jobs_from_log(log_file)
        assert len(failed) == 2
        rules = {j.rule for j in failed}
        assert rules == {"sort", "merge"}

    def test_deduplicates_errors(self, snakemake_dir: Path) -> None:
        """Test that duplicate error messages are deduplicated."""
        log_content = """
rule align:
    jobid: 1
Error in rule align:
    First error occurrence
Error in rule align:
    Second error occurrence (summary)
"""
        log_file = snakemake_dir / "log" / "test.snakemake.log"
        log_file.write_text(log_content)

        failed = parse_failed_jobs_from_log(log_file)
        # Should only have one entry for align, not two
        assert len(failed) == 1
        assert failed[0].rule == "align"

    def test_nonexistent_file(self, snakemake_dir: Path) -> None:
        """Test with nonexistent log file."""
        log_file = snakemake_dir / "log" / "nonexistent.log"
        failed = parse_failed_jobs_from_log(log_file)
        assert len(failed) == 0


class TestParseWorkflowState:
    """Tests for parse_workflow_state function."""

    def test_nonexistent_snakemake_dir(self, tmp_path: Path) -> None:
        """Test with missing .snakemake directory."""
        state = parse_workflow_state(tmp_path)
        assert state.status == WorkflowStatus.COMPLETED
        assert state.total_jobs == 0

    def test_running_workflow(self, snakemake_dir: Path, tmp_path: Path) -> None:
        """Test detecting a running workflow."""
        # Create lock file
        (snakemake_dir / "locks" / "0.input.lock").write_text("/file")

        # Create log with progress
        log_file = snakemake_dir / "log" / "2024-01-01T120000.snakemake.log"
        log_file.write_text("5 of 10 steps (50%) done")

        state = parse_workflow_state(tmp_path)
        assert state.status == WorkflowStatus.RUNNING
        assert state.total_jobs == 10
        assert state.completed_jobs == 5

    def test_completed_workflow(self, snakemake_dir: Path, tmp_path: Path) -> None:
        """Test detecting a completed workflow."""
        # No lock files, progress shows complete
        log_file = snakemake_dir / "log" / "2024-01-01T120000.snakemake.log"
        log_file.write_text("10 of 10 steps (100%) done")

        state = parse_workflow_state(tmp_path)
        assert state.status == WorkflowStatus.COMPLETED
        assert state.total_jobs == 10
        assert state.completed_jobs == 10

    def test_incomplete_workflow(self, snakemake_dir: Path, tmp_path: Path) -> None:
        """Test detecting an incomplete (interrupted) workflow."""
        # No lock files, but progress incomplete - workflow was interrupted
        log_file = snakemake_dir / "log" / "2024-01-01T120000.snakemake.log"
        log_file.write_text("5 of 10 steps (50%) done")

        state = parse_workflow_state(tmp_path)
        assert state.status == WorkflowStatus.INCOMPLETE
        assert state.failed_jobs == 0  # No explicit failures, just interrupted


def _local_timestamp(timestamp_str: str) -> float:
    """Parse timestamp string to Unix time using local timezone (same as parser)."""
    from datetime import datetime

    dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y")
    return dt.timestamp()


class TestAugmentCompletionsWithThreads:
    """Tests for _augment_completions_with_threads function."""

    def test_augments_missing_threads(self, tmp_path: Path) -> None:
        """Test that threads are added to completions from log data."""
        from snakesee.models import JobInfo
        from snakesee.parser import _augment_completions_with_threads

        # Use timestamps that will be parsed consistently in any timezone
        start_ts_str = "Fri Jan  1 12:00:00 2024"
        end_ts_str = "Fri Jan  1 12:01:00 2024"
        start_time = _local_timestamp(start_ts_str)
        end_time = _local_timestamp(end_ts_str)

        # Create log file with thread info
        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "2024-01-01T120000.snakemake.log"
        log_file.write_text(
            f"[{start_ts_str}]\n"
            "rule align:\n"
            "    output: out.bam\n"
            "    jobid: 1\n"
            "    threads: 4\n"
            f"[{end_ts_str}]\n"
            "Finished job 1.\n"
        )

        # Create completion without threads - end_time matches log finish timestamp
        completions = [
            JobInfo(
                rule="align",
                job_id="1",
                start_time=start_time,
                end_time=end_time,
            )
        ]

        result = _augment_completions_with_threads(completions, log_file)
        assert len(result) == 1
        assert result[0].threads == 4

    def test_preserves_existing_threads(self, tmp_path: Path) -> None:
        """Test that existing thread values are not overwritten."""
        from snakesee.models import JobInfo
        from snakesee.parser import _augment_completions_with_threads

        # Use timestamps that will be parsed consistently in any timezone
        start_ts_str = "Fri Jan  1 12:00:00 2024"
        end_ts_str = "Fri Jan  1 12:01:00 2024"
        start_time = _local_timestamp(start_ts_str)
        end_time = _local_timestamp(end_ts_str)

        # Create log file with different thread info
        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "2024-01-01T120000.snakemake.log"
        log_file.write_text(
            f"[{start_ts_str}]\n"
            "rule align:\n"
            "    output: out.bam\n"
            "    jobid: 1\n"
            "    threads: 8\n"  # Different from existing
            f"[{end_ts_str}]\n"
            "Finished job 1.\n"
        )

        # Create completion WITH existing threads
        completions = [
            JobInfo(
                rule="align",
                job_id="1",
                start_time=start_time,
                end_time=end_time,
                threads=4,  # Already has threads
            )
        ]

        result = _augment_completions_with_threads(completions, log_file)
        assert len(result) == 1
        assert result[0].threads == 4  # Should preserve original, not overwrite with 8

    def test_no_match_returns_original(self, tmp_path: Path) -> None:
        """Test that unmatched completions are returned unchanged."""
        from snakesee.models import JobInfo
        from snakesee.parser import _augment_completions_with_threads

        # Create empty log file
        log_dir = tmp_path / ".snakemake" / "log"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "2024-01-01T120000.snakemake.log"
        log_file.write_text("")

        # Use a timestamp that won't match anything (different from log)
        completions = [JobInfo(rule="align", job_id="1", end_time=1704110460.0)]

        result = _augment_completions_with_threads(completions, log_file)
        assert len(result) == 1
        assert result[0].threads is None


class TestCollectWildcardTimingStats:
    """Tests for collect_wildcard_timing_stats function."""

    def test_empty_metadata(self, tmp_path: Path) -> None:
        """Test with empty metadata directory."""
        meta_dir = tmp_path / ".snakemake" / "metadata"
        meta_dir.mkdir(parents=True)

        result = collect_wildcard_timing_stats(meta_dir)
        assert result == {}

    def test_metadata_without_wildcards(self, tmp_path: Path) -> None:
        """Test metadata without wildcards returns empty."""
        meta_dir = tmp_path / ".snakemake" / "metadata"
        meta_dir.mkdir(parents=True)

        # Create metadata without wildcards
        metadata = {
            "rule": "align",
            "starttime": 1000.0,
            "endtime": 1100.0,
        }
        (meta_dir / "align_0").write_text(json.dumps(metadata))

        result = collect_wildcard_timing_stats(meta_dir)
        assert result == {}

    def test_metadata_with_wildcards(self, tmp_path: Path) -> None:
        """Test metadata with wildcards is collected."""
        meta_dir = tmp_path / ".snakemake" / "metadata"
        meta_dir.mkdir(parents=True)

        # Create metadata with wildcards
        for i, sample in enumerate(["A", "B", "A"]):
            metadata = {
                "rule": "align",
                "starttime": 1000.0 + i * 100,
                "endtime": 1100.0 + i * 100 if sample == "A" else 1150.0 + i * 100,
                "wildcards": {"sample": sample},
            }
            (meta_dir / f"align_{i}").write_text(json.dumps(metadata))

        result = collect_wildcard_timing_stats(meta_dir)

        assert "align" in result
        assert "sample" in result["align"]
        wts = result["align"]["sample"]
        assert wts.rule == "align"
        assert wts.wildcard_key == "sample"
        assert "A" in wts.stats_by_value
        assert "B" in wts.stats_by_value
        assert wts.stats_by_value["A"].count == 2
        assert wts.stats_by_value["B"].count == 1


class TestIncrementalLogReader:
    """Tests for IncrementalLogReader class."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test reading an empty log file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("")

        reader = IncrementalLogReader(log_file)
        lines = reader.read_new_lines()

        assert lines == 0
        assert reader.progress == (0, 0)
        assert reader.running_jobs == []
        assert reader.completed_jobs == []
        assert reader.failed_jobs == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading from nonexistent file."""
        log_file = tmp_path / "nonexistent.log"
        reader = IncrementalLogReader(log_file)

        lines = reader.read_new_lines()
        assert lines == 0

    def test_progress_parsing(self, tmp_path: Path) -> None:
        """Test parsing progress lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text("5 of 10 steps (50%) done\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        assert reader.progress == (5, 10)

    def test_progress_updates(self, tmp_path: Path) -> None:
        """Test that progress updates with new lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text("1 of 10 steps (10%) done\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()
        assert reader.progress == (1, 10)

        # Append more content
        with open(log_file, "a") as f:
            f.write("5 of 10 steps (50%) done\n")

        reader.read_new_lines()
        assert reader.progress == (5, 10)

    def test_running_jobs(self, tmp_path: Path) -> None:
        """Test detecting running jobs."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""rule align:
    wildcards: sample=A
    jobid: 1
rule sort:
    jobid: 2
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        running = reader.running_jobs
        assert len(running) == 2
        rules = {j.rule for j in running}
        assert rules == {"align", "sort"}

        # Check wildcards captured
        align_job = next(j for j in running if j.rule == "align")
        assert align_job.wildcards == {"sample": "A"}

    def test_completed_jobs(self, tmp_path: Path) -> None:
        """Test detecting completed jobs."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""[Mon Dec 16 10:00:00 2024]
rule align:
    wildcards: sample=A
    jobid: 1
[Mon Dec 16 10:01:00 2024]
Finished job 1.
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Job is completed, not running
        assert len(reader.running_jobs) == 0
        assert len(reader.completed_jobs) == 1

        completed = reader.completed_jobs[0]
        assert completed.rule == "align"
        assert completed.job_id == "1"
        assert completed.wildcards == {"sample": "A"}

    def test_failed_jobs(self, tmp_path: Path) -> None:
        """Test detecting failed jobs."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""rule align:
    jobid: 1
Error in rule align:
    Some error
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        failed = reader.failed_jobs
        assert len(failed) == 1
        assert failed[0].rule == "align"
        assert failed[0].job_id == "1"

    def test_incremental_reading(self, tmp_path: Path) -> None:
        """Test that only new lines are parsed on subsequent calls."""
        log_file = tmp_path / "test.log"
        log_file.write_text("rule align:\n    jobid: 1\n")

        reader = IncrementalLogReader(log_file)
        lines1 = reader.read_new_lines()
        assert lines1 == 2
        assert len(reader.running_jobs) == 1

        # Append more content
        with open(log_file, "a") as f:
            f.write("Finished job 1.\n")

        lines2 = reader.read_new_lines()
        assert lines2 == 1  # Only the new line

        # Job should now be completed, not running
        assert len(reader.running_jobs) == 0
        assert len(reader.completed_jobs) == 1

    def test_reset(self, tmp_path: Path) -> None:
        """Test reset clears all state."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""rule align:
    jobid: 1
5 of 10 steps (50%) done
Error in rule align:
    Error
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        assert reader.progress == (5, 10)
        assert len(reader.running_jobs) == 1
        assert len(reader.failed_jobs) == 1

        reader.reset()

        assert reader.progress == (0, 0)
        assert reader.running_jobs == []
        assert reader.failed_jobs == []

    def test_set_log_path_different_file(self, tmp_path: Path) -> None:
        """Test that changing log path resets state."""
        log1 = tmp_path / "log1.log"
        log2 = tmp_path / "log2.log"
        log1.write_text("5 of 10 steps (50%) done\n")
        log2.write_text("3 of 20 steps (15%) done\n")

        reader = IncrementalLogReader(log1)
        reader.read_new_lines()
        assert reader.progress == (5, 10)

        reader.set_log_path(log2)
        reader.read_new_lines()
        assert reader.progress == (3, 20)

    def test_set_log_path_same_file(self, tmp_path: Path) -> None:
        """Test that setting same path doesn't reset state."""
        log_file = tmp_path / "test.log"
        log_file.write_text("rule align:\n    jobid: 1\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()
        assert len(reader.running_jobs) == 1

        # Set same path - should not reset
        reader.set_log_path(log_file)
        assert len(reader.running_jobs) == 1

    def test_file_rotation_detection(self, tmp_path: Path) -> None:
        """Test that file rotation (truncation) resets state."""
        log_file = tmp_path / "test.log"
        log_file.write_text("rule align:\n    jobid: 1\n5 of 10 steps (50%) done\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()
        assert reader.progress == (5, 10)

        # Simulate rotation by truncating and writing new content
        log_file.write_text("1 of 5 steps (20%) done\n")

        reader.read_new_lines()
        # State should be reset and new content parsed
        assert reader.progress == (1, 5)
        assert len(reader.running_jobs) == 0

    def test_deduplicates_failed_jobs(self, tmp_path: Path) -> None:
        """Test that duplicate error messages don't create duplicate entries."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""rule align:
    jobid: 1
Error in rule align:
    First error
Error in rule align:
    Second error (same rule, same job)
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Should only have one failed job entry
        assert len(reader.failed_jobs) == 1

    def test_completed_jobs_sorted_newest_first(self, tmp_path: Path) -> None:
        """Test that completed jobs are sorted by end time, newest first."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""[Mon Dec 16 10:00:00 2024]
rule first:
    jobid: 1
[Mon Dec 16 10:01:00 2024]
Finished job 1.
[Mon Dec 16 10:02:00 2024]
rule second:
    jobid: 2
[Mon Dec 16 10:03:00 2024]
Finished job 2.
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        completed = reader.completed_jobs
        assert len(completed) == 2
        # Newest first
        assert completed[0].rule == "second"
        assert completed[1].rule == "first"


class TestParsePositiveInt:
    """Tests for _parse_positive_int function."""

    def test_valid_positive_int(self) -> None:
        """Test parsing valid positive integers."""
        from snakesee.parser import _parse_positive_int

        assert _parse_positive_int("1") == 1
        assert _parse_positive_int("42") == 42
        assert _parse_positive_int("1000") == 1000

    def test_invalid_not_a_number(self) -> None:
        """Test parsing non-numeric strings."""
        from snakesee.parser import _parse_positive_int

        assert _parse_positive_int("not_a_number") is None
        assert _parse_positive_int("abc") is None
        assert _parse_positive_int("") is None

    def test_invalid_zero(self) -> None:
        """Test parsing zero (not positive)."""
        from snakesee.parser import _parse_positive_int

        assert _parse_positive_int("0") is None

    def test_invalid_negative(self) -> None:
        """Test parsing negative numbers."""
        from snakesee.parser import _parse_positive_int

        assert _parse_positive_int("-1") is None
        assert _parse_positive_int("-100") is None


class TestParseNonNegativeInt:
    """Tests for _parse_non_negative_int function."""

    def test_valid_non_negative_int(self) -> None:
        """Test parsing valid non-negative integers."""
        from snakesee.parser import _parse_non_negative_int

        assert _parse_non_negative_int("0") == 0
        assert _parse_non_negative_int("1") == 1
        assert _parse_non_negative_int("42") == 42

    def test_invalid_not_a_number(self) -> None:
        """Test parsing non-numeric strings."""
        from snakesee.parser import _parse_non_negative_int

        assert _parse_non_negative_int("not_a_number") is None
        assert _parse_non_negative_int("abc") is None

    def test_invalid_negative(self) -> None:
        """Test parsing negative numbers."""
        from snakesee.parser import _parse_non_negative_int

        assert _parse_non_negative_int("-1") is None
        assert _parse_non_negative_int("-100") is None


class TestSafeMtime:
    """Tests for safe_mtime function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Test getting mtime of existing file."""
        from snakesee.utils import safe_mtime

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        mtime = safe_mtime(test_file)
        assert mtime > 0

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test getting mtime of nonexistent file returns 0."""
        from snakesee.utils import safe_mtime

        nonexistent = tmp_path / "does_not_exist.txt"
        assert safe_mtime(nonexistent) == 0


class TestParseJobStatsFromLog:
    """Tests for parse_job_stats_from_log function."""

    def test_valid_job_stats_table(self, tmp_path: Path) -> None:
        """Test parsing a valid job stats table."""
        from snakesee.parser import parse_job_stats_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("""Building DAG of jobs...
Job stats:
job        count
-------  -------
align          5
sort           5
index          5
all            1
total         16
""")
        rules = parse_job_stats_from_log(log_file)
        # Note: parser skips "total" row as it's a summary, not a rule
        assert rules == {"align", "sort", "index", "all"}

    def test_no_job_stats_section(self, tmp_path: Path) -> None:
        """Test log without job stats section returns empty set."""
        from snakesee.parser import parse_job_stats_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("""Some other log content
rule align:
    jobid: 1
""")
        rules = parse_job_stats_from_log(log_file)
        assert rules == set()

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test nonexistent file returns empty set."""
        from snakesee.parser import parse_job_stats_from_log

        nonexistent = tmp_path / "does_not_exist.log"
        rules = parse_job_stats_from_log(nonexistent)
        assert rules == set()


class TestParseJobStatsCountsFromLog:
    """Tests for parse_job_stats_counts_from_log function."""

    def test_valid_counts(self, tmp_path: Path) -> None:
        """Test parsing valid job counts."""
        from snakesee.parser import parse_job_stats_counts_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("""Job stats:
job        count
-------  -------
align          5
sort           3
all            1
total          9
""")
        counts = parse_job_stats_counts_from_log(log_file)
        assert counts["align"] == 5
        assert counts["sort"] == 3
        assert counts["all"] == 1
        # Note: "total" row is skipped by the parser
        assert "total" not in counts

    def test_no_job_stats(self, tmp_path: Path) -> None:
        """Test log without job stats returns empty dict."""
        from snakesee.parser import parse_job_stats_counts_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("Some log content without job stats\n")
        counts = parse_job_stats_counts_from_log(log_file)
        assert counts == {}


class TestParseIncompleteJobs:
    """Tests for parse_incomplete_jobs function."""

    def test_no_incomplete_dir(self, tmp_path: Path) -> None:
        """Test when incomplete directory doesn't exist."""
        from snakesee.parser import parse_incomplete_jobs

        snakemake_dir = tmp_path / ".snakemake"
        snakemake_dir.mkdir()
        incomplete_dir = snakemake_dir / "incomplete"
        # No incomplete directory created
        result = list(parse_incomplete_jobs(incomplete_dir))
        assert result == []

    def test_empty_incomplete_dir(self, tmp_path: Path) -> None:
        """Test when incomplete directory is empty."""
        from snakesee.parser import parse_incomplete_jobs

        snakemake_dir = tmp_path / ".snakemake"
        incomplete_dir = snakemake_dir / "incomplete"
        incomplete_dir.mkdir(parents=True)
        result = list(parse_incomplete_jobs(incomplete_dir))
        assert result == []

    def test_invalid_marker_files(self, tmp_path: Path) -> None:
        """Test that invalid marker files are skipped gracefully."""
        from snakesee.parser import parse_incomplete_jobs

        snakemake_dir = tmp_path / ".snakemake"
        incomplete_dir = snakemake_dir / "incomplete"
        incomplete_dir.mkdir(parents=True)

        # Create a file that's not a valid base64-encoded path
        (incomplete_dir / "not_base64!!!").touch()

        # Should not crash, just return empty or skip invalid
        result = list(parse_incomplete_jobs(incomplete_dir))
        # May return empty or have parsing issues, but shouldn't crash
        assert isinstance(result, list)


class TestIncrementalLogReaderEdgeCases:
    """Additional edge case tests for IncrementalLogReader."""

    def test_threads_with_invalid_value(self, tmp_path: Path) -> None:
        """Test handling of invalid thread values."""
        log_file = tmp_path / "test.log"
        log_file.write_text("""rule align:
    jobid: 1
    threads: not_a_number
""")
        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Should still parse the job, just without valid threads
        assert len(reader.running_jobs) == 1
        # Threads should be None or default when invalid
        job = reader.running_jobs[0]
        assert job.threads is None or job.threads == 1

    def test_log_file_deleted_during_read(self, tmp_path: Path) -> None:
        """Test handling when log file is deleted after opening."""
        log_file = tmp_path / "test.log"
        log_file.write_text("initial content\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Delete the file
        log_file.unlink()

        # Should handle gracefully on next read
        reader.read_new_lines()  # Should not crash

    def test_empty_log_file(self, tmp_path: Path) -> None:
        """Test handling of empty log file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        assert reader.progress == (0, 0)
        assert len(reader.running_jobs) == 0
        assert len(reader.failed_jobs) == 0

    def test_binary_content_in_log(self, tmp_path: Path) -> None:
        """Test handling of binary/non-UTF8 content in log."""
        log_file = tmp_path / "test.log"
        # Write some valid content with binary garbage
        content = b"rule align:\n    jobid: 1\n\xff\xfe\x00\x01\nFinished job 1.\n"
        log_file.write_bytes(content)

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()  # Should not crash

    def test_very_long_lines(self, tmp_path: Path) -> None:
        """Test handling of very long lines."""
        log_file = tmp_path / "test.log"
        long_line = "x" * 100000  # 100KB line
        log_file.write_text(f"rule align:\n    jobid: 1\n{long_line}\nFinished job 1.\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()  # Should not crash or hang


class TestParseTimestamp:
    """Tests for _parse_timestamp function."""

    def test_valid_timestamp(self) -> None:
        """Test parsing valid timestamp."""
        from snakesee.parser import _parse_timestamp

        ts = _parse_timestamp("Mon Dec 16 10:00:00 2024")
        assert ts is not None
        assert ts > 0

    def test_invalid_timestamp(self) -> None:
        """Test parsing invalid timestamp returns None."""
        from snakesee.parser import _parse_timestamp

        assert _parse_timestamp("not a timestamp") is None
        assert _parse_timestamp("") is None
        assert _parse_timestamp("2024-12-16") is None  # Wrong format


class TestMetadataRecord:
    """Tests for MetadataRecord dataclass."""

    def test_duration_with_both_times(self) -> None:
        """Test duration calculation with both start and end times."""
        from snakesee.parser import MetadataRecord

        record = MetadataRecord(
            rule="align",
            start_time=1000.0,
            end_time=1060.0,
        )
        assert record.duration == 60.0

    def test_duration_missing_start_time(self) -> None:
        """Test duration returns None when start_time is missing."""
        from snakesee.parser import MetadataRecord

        record = MetadataRecord(
            rule="align",
            start_time=None,
            end_time=1060.0,
        )
        assert record.duration is None

    def test_duration_missing_end_time(self) -> None:
        """Test duration returns None when end_time is missing."""
        from snakesee.parser import MetadataRecord

        record = MetadataRecord(
            rule="align",
            start_time=1000.0,
            end_time=None,
        )
        assert record.duration is None

    def test_to_job_info(self) -> None:
        """Test conversion to JobInfo."""
        from snakesee.parser import MetadataRecord

        record = MetadataRecord(
            rule="align",
            start_time=1000.0,
            end_time=1060.0,
            wildcards={"sample": "A"},
        )
        job_info = record.to_job_info()
        assert job_info.rule == "align"
        assert job_info.wildcards == {"sample": "A"}
        assert job_info.duration == 60.0


class TestCollectRuleCodeHashes:
    """Tests for collect_rule_code_hashes function.

    Note: collect_rule_code_hashes returns dict[str, set[str]] where
    keys are code hashes and values are sets of rule names that have that hash.
    """

    def test_valid_metadata(self, tmp_path: Path) -> None:
        """Test collecting code hashes from valid metadata."""
        from snakesee.parser import collect_rule_code_hashes

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create a valid metadata file
        metadata = {
            "rule": "align",
            "code": "def align(): pass",
        }
        (metadata_dir / "output.txt").write_text(json.dumps(metadata))

        hashes = collect_rule_code_hashes(metadata_dir)
        # Returns hash -> set of rules, so check if "align" is in any value set
        all_rules = set()
        for rules in hashes.values():
            all_rules.update(rules)
        assert "align" in all_rules

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Test handling of invalid JSON in metadata."""
        from snakesee.parser import collect_rule_code_hashes

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create invalid JSON
        (metadata_dir / "output.txt").write_text("not valid json {{{")

        # Should not crash
        hashes = collect_rule_code_hashes(metadata_dir)
        assert isinstance(hashes, dict)

    def test_missing_rule_key(self, tmp_path: Path) -> None:
        """Test handling of metadata missing 'rule' key."""
        from snakesee.parser import collect_rule_code_hashes

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create metadata without 'rule' key
        (metadata_dir / "output.txt").write_text(json.dumps({"code": "some code"}))

        # Should not crash
        hashes = collect_rule_code_hashes(metadata_dir)
        assert isinstance(hashes, dict)


class TestDetermineFinalWorkflowStatus:
    """Tests for _determine_final_workflow_status helper."""

    def test_running_with_active_jobs(self) -> None:
        """Running jobs on latest log with running workflow -> RUNNING."""
        from snakesee.models import JobInfo
        from snakesee.parser.core import _determine_final_workflow_status

        running = [JobInfo(rule="test", job_id="1")]
        status = _determine_final_workflow_status(
            running=running,
            failed_list=[],
            incomplete_list=[],
            completed=5,
            total=10,
            is_latest_log=True,
            workflow_is_running=True,
        )
        assert status == WorkflowStatus.RUNNING

    def test_failed_with_failures(self) -> None:
        """Failed jobs present -> FAILED."""
        from snakesee.models import JobInfo
        from snakesee.parser.core import _determine_final_workflow_status

        failed = [JobInfo(rule="test", job_id="1")]
        status = _determine_final_workflow_status(
            running=[],
            failed_list=failed,
            incomplete_list=[],
            completed=5,
            total=10,
            is_latest_log=True,
            workflow_is_running=False,
        )
        assert status == WorkflowStatus.FAILED

    def test_incomplete_with_markers(self) -> None:
        """Incomplete markers present -> INCOMPLETE."""
        from snakesee.models import JobInfo
        from snakesee.parser.core import _determine_final_workflow_status

        incomplete = [JobInfo(rule="test", job_id="1")]
        status = _determine_final_workflow_status(
            running=[],
            failed_list=[],
            incomplete_list=incomplete,
            completed=5,
            total=10,
            is_latest_log=True,
            workflow_is_running=False,
        )
        assert status == WorkflowStatus.INCOMPLETE

    def test_incomplete_partial_completion(self) -> None:
        """Not all jobs done and not running -> INCOMPLETE."""
        from snakesee.parser.core import _determine_final_workflow_status

        status = _determine_final_workflow_status(
            running=[],
            failed_list=[],
            incomplete_list=[],
            completed=5,
            total=10,
            is_latest_log=True,
            workflow_is_running=False,
        )
        assert status == WorkflowStatus.INCOMPLETE

    def test_completed_all_done(self) -> None:
        """All jobs completed, not running -> COMPLETED."""
        from snakesee.parser.core import _determine_final_workflow_status

        status = _determine_final_workflow_status(
            running=[],
            failed_list=[],
            incomplete_list=[],
            completed=10,
            total=10,
            is_latest_log=True,
            workflow_is_running=False,
        )
        assert status == WorkflowStatus.COMPLETED


class TestReconcileJobLists:
    """Tests for _reconcile_job_lists helper."""

    def test_removes_failed_from_running(self) -> None:
        """Failed jobs should be removed from running list."""
        from snakesee.models import JobInfo
        from snakesee.parser.core import _reconcile_job_lists

        running = [
            JobInfo(rule="test1", job_id="1"),
            JobInfo(rule="test2", job_id="2"),
        ]
        failed = [JobInfo(rule="test1", job_id="1")]
        incomplete: list[JobInfo] = []

        new_running, new_incomplete = _reconcile_job_lists(
            running, failed, incomplete, workflow_is_running=True
        )

        assert len(new_running) == 1
        assert new_running[0].job_id == "2"
        assert new_incomplete == []

    def test_clears_running_when_not_running(self) -> None:
        """When workflow not running, running list should be cleared."""
        from snakesee.models import JobInfo
        from snakesee.parser.core import _reconcile_job_lists

        running = [JobInfo(rule="test", job_id="1")]
        incomplete = [JobInfo(rule="test2", job_id="2")]

        new_running, new_incomplete = _reconcile_job_lists(
            running, [], incomplete, workflow_is_running=False
        )

        assert new_running == []
        assert new_incomplete == incomplete

    def test_clears_incomplete_when_running(self) -> None:
        """When workflow is running, incomplete markers should be cleared."""
        from snakesee.models import JobInfo
        from snakesee.parser.core import _reconcile_job_lists

        running = [JobInfo(rule="test", job_id="1")]
        incomplete = [JobInfo(rule="test2", job_id="2")]

        new_running, new_incomplete = _reconcile_job_lists(
            running, [], incomplete, workflow_is_running=True
        )

        assert new_running == running
        assert new_incomplete == []


class TestCollectFilteredCompletions:
    """Tests for _collect_filtered_completions helper."""

    def test_returns_empty_for_no_log(self) -> None:
        """Returns empty list when no log path."""
        from snakesee.parser.core import _collect_filtered_completions

        result = _collect_filtered_completions(
            all_completions=[],
            log_path=None,
            cutoff_time=None,
            log_reader=None,
        )
        assert result == []

    def test_uses_log_reader_completions(self, tmp_path: Path) -> None:
        """Uses log reader completions when available and no metadata match."""
        from snakesee.parser import IncrementalLogReader
        from snakesee.parser.core import _collect_filtered_completions

        # Create a log file with a completed job
        log_file = tmp_path / "test.log"
        log_file.write_text("""[Mon Dec 16 10:00:00 2024]
rule test:
    jobid: 1
[Mon Dec 16 10:01:00 2024]
Finished job 1.
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Call with empty all_completions to force reader fallback
        result = _collect_filtered_completions(
            all_completions=[],
            log_path=log_file,
            cutoff_time=None,
            log_reader=reader,
        )

        assert len(result) == 1
        assert result[0].rule == "test"


class TestParserEdgeCasesUnit:
    """Unit tests for parser edge cases with special characters and malformed data."""

    def test_parse_wildcards_with_special_chars(self) -> None:
        """Test parsing wildcards with special characters."""
        from snakesee.parser import _parse_wildcards

        # Path-like value
        result = _parse_wildcards("sample=path/to/file.txt")
        assert result == {"sample": "path/to/file.txt"}

        # Dots in value
        result = _parse_wildcards("sample=file.name.txt")
        assert result == {"sample": "file.name.txt"}

        # Hyphens and underscores
        result = _parse_wildcards("sample_name=test-value_123")
        assert result == {"sample_name": "test-value_123"}

    def test_parse_wildcards_with_equals_in_value(self) -> None:
        """Test parsing wildcards with = in value."""
        from snakesee.parser import _parse_wildcards

        # This is a tricky case - = in value
        result = _parse_wildcards("param=key=value")
        # Should capture at least the key
        assert "param" in result

    def test_empty_log_file(self, tmp_path: Path) -> None:
        """Test parsing empty log file."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("")

        reader = IncrementalLogReader(log_file)
        lines = reader.read_new_lines()

        assert lines == 0
        assert reader.progress == (0, 0)
        assert reader.running_jobs == []

    def test_log_file_with_only_whitespace(self, tmp_path: Path) -> None:
        """Test parsing log file with only whitespace."""
        log_file = tmp_path / "whitespace.log"
        log_file.write_text("   \n\t\n  \n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        assert reader.progress == (0, 0)
        assert reader.running_jobs == []

    def test_log_with_unicode_characters(self, tmp_path: Path) -> None:
        """Test parsing log with unicode characters."""
        log_file = tmp_path / "unicode.log"
        log_file.write_text("""rule process_日本語:
    jobid: 1
[Mon Dec 16 10:00:00 2024]
Finished job 1.
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Should handle unicode rule names
        completed = reader.completed_jobs
        assert len(completed) == 1
        assert "日本語" in completed[0].rule

    def test_log_with_very_long_line(self, tmp_path: Path) -> None:
        """Test parsing log with very long lines."""
        log_file = tmp_path / "long.log"
        long_value = "x" * 10000
        log_file.write_text(f"""rule long_rule:
    wildcards: sample={long_value}
    jobid: 1
""")

        reader = IncrementalLogReader(log_file)
        lines = reader.read_new_lines()

        # Should handle long lines without crashing
        assert lines > 0
        running = reader.running_jobs
        assert len(running) == 1

    def test_log_with_malformed_timestamp(self, tmp_path: Path) -> None:
        """Test parsing log with malformed timestamps."""
        log_file = tmp_path / "malformed_time.log"
        log_file.write_text("""[Invalid Timestamp Format]
rule test:
    jobid: 1
[Another Bad 00:00:00 2024]
Finished job 1.
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Should still parse the job info - job completed
        assert len(reader.running_jobs) == 0
        assert len(reader.completed_jobs) == 1

    def test_log_with_interleaved_output(self, tmp_path: Path) -> None:
        """Test parsing log with interleaved rule outputs."""
        log_file = tmp_path / "interleaved.log"
        log_file.write_text("""rule align:
    jobid: 1
rule sort:
    jobid: 2
    wildcards: sample=A
rule align:
    wildcards: sample=B
Finished job 1.
Finished job 2.
""")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        # Both jobs should be completed
        completed = reader.completed_jobs
        assert len(completed) == 2

    def test_parse_progress_with_large_numbers(self, tmp_path: Path) -> None:
        """Test parsing progress with large job counts."""
        log_file = tmp_path / "large.log"
        log_file.write_text("99999 of 100000 steps (99%) done\n")

        reader = IncrementalLogReader(log_file)
        reader.read_new_lines()

        assert reader.progress == (99999, 100000)

    def test_log_with_null_bytes(self, tmp_path: Path) -> None:
        """Test handling log with null bytes (binary data)."""
        log_file = tmp_path / "binary.log"
        # Write text with embedded null bytes
        content = "rule test:\n    jobid: 1\x00\n"
        log_file.write_bytes(content.encode("utf-8", errors="replace"))

        reader = IncrementalLogReader(log_file)
        # Should not crash
        try:
            reader.read_new_lines()
        except UnicodeDecodeError:
            pass  # Acceptable to fail on binary data

    def test_parse_metadata_with_missing_fields(self, tmp_path: Path) -> None:
        """Test parsing metadata files with missing required fields are skipped."""
        from snakesee.parser import parse_metadata_files

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Metadata with only rule name, missing starttime and endtime
        (metadata_dir / "output.txt").write_text(json.dumps({"rule": "test"}))

        # Files missing required fields (rule, starttime, endtime) are silently skipped
        jobs = list(parse_metadata_files(metadata_dir))
        assert len(jobs) == 0

    def test_parse_metadata_with_extra_fields(self, tmp_path: Path) -> None:
        """Test parsing metadata files with extra unknown fields."""
        from snakesee.parser import parse_metadata_files

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Metadata with required fields plus extra unknown fields
        (metadata_dir / "output.txt").write_text(
            json.dumps(
                {
                    "rule": "test",
                    "starttime": 900.0,
                    "endtime": 1000.0,
                    "unknown_field": "value",
                    "another_extra": 123,
                }
            )
        )

        # Extra fields should be ignored, required fields parsed
        jobs = list(parse_metadata_files(metadata_dir))
        assert len(jobs) == 1
        assert jobs[0].rule == "test"
        assert jobs[0].start_time == 900.0
        assert jobs[0].end_time == 1000.0


# =============================================================================
# Property-Based Tests (TC-L2)
# =============================================================================


class TestParserPropertyBased:
    """Property-based tests using hypothesis for parser robustness."""

    # Strategy for valid wildcard keys (alphanumeric + underscore)
    wildcard_key = st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
        min_size=1,
        max_size=30,
    )
    # Strategy for valid wildcard values (alphanumeric + common chars, no comma)
    wildcard_value = st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-."),
        min_size=1,
        max_size=50,
    )

    @given(key=wildcard_key, value=wildcard_value)
    def test_parse_wildcards_roundtrip(self, key: str, value: str) -> None:
        """Wildcard parsing should handle valid key=value pairs."""
        from snakesee.parser import _parse_wildcards

        # Create wildcard string
        wildcard_str = f"{key}={value}"
        result = _parse_wildcards(wildcard_str)

        # Should parse without error
        assert result is not None
        assert key in result
        assert result[key] == value

    @given(
        keys=st.lists(wildcard_key, min_size=1, max_size=5, unique=True),
        values=st.lists(wildcard_value, min_size=1, max_size=5),
    )
    def test_parse_wildcards_multiple(self, keys: list[str], values: list[str]) -> None:
        """Wildcard parsing handles multiple key=value pairs."""
        from snakesee.parser import _parse_wildcards

        # Match lengths
        n = min(len(keys), len(values))
        keys = keys[:n]
        values = values[:n]

        # Create wildcard string
        pairs = [f"{k}={v}" for k, v in zip(keys, values, strict=True)]
        wildcard_str = ", ".join(pairs)
        result = _parse_wildcards(wildcard_str)

        # Should parse all pairs
        assert result is not None
        for k, v in zip(keys, values, strict=True):
            assert k in result
            assert result[k] == v

    @given(
        completed=st.integers(min_value=0, max_value=1000000),
        total=st.integers(min_value=1, max_value=1000000),
    )
    def test_parse_progress_line_valid(self, completed: int, total: int) -> None:
        """Progress parsing handles valid progress lines."""
        from snakesee.parser.line_parser import LogLineParser
        from snakesee.parser.line_parser import ParseEventType

        # Ensure completed <= total for valid progress
        if completed > total:
            completed, total = total, completed

        parser = LogLineParser()
        line = f"{completed} of {total} steps (50%) done"
        event = parser.parse_line(line)

        # Should parse progress
        assert event is not None
        assert event.event_type == ParseEventType.PROGRESS
        assert event.data["completed"] == completed
        assert event.data["total"] == total

    @given(line=st.text(max_size=500))
    def test_parse_line_no_crash(self, line: str) -> None:
        """Line parsing should never crash on arbitrary input."""
        from snakesee.parser.line_parser import LogLineParser

        parser = LogLineParser()
        # Should not crash on any input
        result = parser.parse_line(line)
        # Result is either None or a valid ParseEvent
        assert result is None or hasattr(result, "event_type")

    @given(line=st.text(max_size=1000))
    def test_log_line_parsing_stability(self, line: str) -> None:
        """Log line parsing should maintain valid context on arbitrary input."""
        from snakesee.parser.line_parser import LogLineParser

        parser = LogLineParser()
        # Parse multiple lines - context should remain valid
        parser.parse_line(line)
        parser.parse_line(line)

        # Context should be valid dataclass
        assert parser.context is not None
        assert hasattr(parser.context, "rule")
        assert hasattr(parser.context, "jobid")

    @given(
        rule=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
            min_size=1,
            max_size=30,
        ),
        jobid=st.integers(min_value=0, max_value=999999),
    )
    def test_job_lifecycle_properties(self, rule: str, jobid: int) -> None:
        """Job tracking maintains consistent state."""
        from snakesee.parser.job_tracker import JobLifecycleTracker

        tracker = JobLifecycleTracker()

        # Start a job with correct API
        tracker.start_job(str(jobid), rule, start_time=1000.0)

        # Verify job is tracked
        running = tracker.get_running_jobs()
        running_ids = [j.job_id for j in running]
        assert str(jobid) in running_ids

        # Complete the job
        completed_job = tracker.finish_job(str(jobid), 1100.0)

        # Job should be completed
        assert completed_job is not None
        assert completed_job.rule == rule
        assert completed_job.job_id == str(jobid)

        # Job should no longer be running
        running_after = tracker.get_running_jobs()
        running_ids_after = [j.job_id for j in running_after]
        assert str(jobid) not in running_ids_after

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        rule=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
            min_size=1,
            max_size=30,
        ),
        starttime=st.floats(min_value=0, max_value=1e10, allow_nan=False),
        duration=st.floats(min_value=0.1, max_value=1e6, allow_nan=False),
    )
    def test_metadata_parsing_properties(
        self, rule: str, starttime: float, duration: float
    ) -> None:
        """Metadata parsing handles valid data correctly."""
        import tempfile

        from snakesee.parser import parse_metadata_files

        endtime = starttime + duration
        data = {"rule": rule, "starttime": starttime, "endtime": endtime}

        # Use tempfile to avoid fixture reuse issues
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "test_output").write_text(json.dumps(data))

            jobs = list(parse_metadata_files(metadata_dir))
            assert len(jobs) == 1
            assert jobs[0].rule == rule
            assert jobs[0].start_time == starttime
            assert jobs[0].end_time == endtime


class TestParseAllJobsFromLog:
    """Tests for parse_all_jobs_from_log function."""

    def test_parse_jobs_with_wildcards_and_threads(self, tmp_path: Path) -> None:
        """Test parsing all scheduled jobs with wildcards and threads."""
        from snakesee.parser import parse_all_jobs_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("""Building DAG of jobs...
rule align:
    input: data/sample_A.fastq
    output: results/sample_A.bam
    wildcards: sample=A
    threads: 4
    jobid: 1

rule align:
    input: data/sample_B.fastq
    output: results/sample_B.bam
    wildcards: sample=B
    threads: 8
    jobid: 2

[Mon Dec 16 10:00:00 2024]
Finished job 1.
""")
        jobs = parse_all_jobs_from_log(log_file)

        assert len(jobs) == 2
        # Jobs should be parsed with their wildcards and threads
        job1 = next(j for j in jobs if j.job_id == "1")
        assert job1.rule == "align"
        assert job1.wildcards == {"sample": "A"}
        assert job1.threads == 4

        job2 = next(j for j in jobs if j.job_id == "2")
        assert job2.rule == "align"
        assert job2.wildcards == {"sample": "B"}
        assert job2.threads == 8

    def test_parse_jobs_no_wildcards(self, tmp_path: Path) -> None:
        """Test parsing jobs without wildcards."""
        from snakesee.parser import parse_all_jobs_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("""rule all:
    input: results/final.txt
    jobid: 0

rule process:
    input: data/input.txt
    output: results/output.txt
    threads: 2
    jobid: 1
""")
        jobs = parse_all_jobs_from_log(log_file)

        assert len(jobs) == 2
        job0 = next(j for j in jobs if j.job_id == "0")
        assert job0.rule == "all"
        assert job0.wildcards is None
        assert job0.threads is None

        job1 = next(j for j in jobs if j.job_id == "1")
        assert job1.rule == "process"
        assert job1.threads == 2

    def test_parse_jobs_empty_log(self, tmp_path: Path) -> None:
        """Test parsing empty log file."""
        from snakesee.parser import parse_all_jobs_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("")

        jobs = parse_all_jobs_from_log(log_file)
        assert jobs == []

    def test_parse_jobs_nonexistent_file(self, tmp_path: Path) -> None:
        """Test parsing nonexistent log file."""
        from snakesee.parser import parse_all_jobs_from_log

        log_file = tmp_path / "nonexistent.log"

        jobs = parse_all_jobs_from_log(log_file)
        assert jobs == []

    def test_parse_jobs_deduplicates_by_jobid(self, tmp_path: Path) -> None:
        """Test that duplicate job IDs are deduplicated."""
        from snakesee.parser import parse_all_jobs_from_log

        log_file = tmp_path / "test.log"
        # Same job ID appearing twice (e.g., from log restart)
        log_file.write_text("""rule align:
    wildcards: sample=A
    jobid: 1

rule align:
    wildcards: sample=A
    jobid: 1
""")
        jobs = parse_all_jobs_from_log(log_file)

        assert len(jobs) == 1
        assert jobs[0].job_id == "1"
