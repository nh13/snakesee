"""Tests for monitor parser functions."""

import json
from pathlib import Path

from snakesee.models import WorkflowStatus
from snakesee.parser import IncrementalLogReader
from snakesee.parser import _parse_wildcards
from snakesee.parser import collect_rule_timing_stats
from snakesee.parser import collect_wildcard_timing_stats
from snakesee.parser import find_latest_log
from snakesee.parser import is_workflow_running
from snakesee.parser import parse_failed_jobs_from_log
from snakesee.parser import parse_metadata_files
from snakesee.parser import parse_progress_from_log
from snakesee.parser import parse_running_jobs_from_log
from snakesee.parser import parse_workflow_state


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
    """Tests for find_latest_log function."""

    def test_no_log_dir(self, tmp_path: Path) -> None:
        """Test when log directory doesn't exist."""
        smk_dir = tmp_path / ".snakemake"
        smk_dir.mkdir()
        assert find_latest_log(smk_dir) is None

    def test_empty_log_dir(self, snakemake_dir: Path) -> None:
        """Test when log directory is empty."""
        assert find_latest_log(snakemake_dir) is None

    def test_single_log(self, snakemake_dir: Path) -> None:
        """Test with a single log file."""
        log_file = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        log_file.write_text("test")
        assert find_latest_log(snakemake_dir) == log_file

    def test_multiple_logs(self, snakemake_dir: Path) -> None:
        """Test returns most recent log."""
        import time

        old_log = snakemake_dir / "log" / "2024-01-01T120000.000000.snakemake.log"
        old_log.write_text("old")
        time.sleep(0.1)

        new_log = snakemake_dir / "log" / "2024-01-02T120000.000000.snakemake.log"
        new_log.write_text("new")

        assert find_latest_log(snakemake_dir) == new_log


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

        # Incomplete markers do NOT indicate the workflow is running - they persist
        # after a workflow is killed. Log freshness is the primary indicator.
        assert is_workflow_running(snakemake_dir) is False


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
