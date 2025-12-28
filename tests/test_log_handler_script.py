"""Tests for log_handler_script.py - Snakemake 8.x log handler integration."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from typing import TextIO
from typing import cast

import pytest

from snakesee import log_handler_script
from snakesee.log_handler_script import EVENT_FILE_NAME
from snakesee.log_handler_script import _extract_resources
from snakesee.log_handler_script import _extract_wildcards
from snakesee.log_handler_script import _handle_job_error
from snakesee.log_handler_script import _handle_job_finished
from snakesee.log_handler_script import _handle_job_info
from snakesee.log_handler_script import _handle_job_started
from snakesee.log_handler_script import _handle_progress
from snakesee.log_handler_script import log_handler


@pytest.fixture
def temp_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Set up a temporary working directory and reset global state."""
    monkeypatch.chdir(tmp_path)

    # Reset global state before each test
    log_handler_script._job_start_times.clear()
    log_handler_script._job_rules.clear()
    log_handler_script._job_wildcards.clear()
    log_handler_script._job_threads.clear()
    log_handler_script._workflow_started_emitted = False
    if log_handler_script._event_file is not None:
        try:
            log_handler_script._event_file.close()
        except OSError:
            pass
        log_handler_script._event_file = None

    yield tmp_path

    # Cleanup after test - use cast to break mypy's narrow analysis
    # (mypy thinks _event_file is always None here because we set it before yield)
    log_handler_script._workflow_started_emitted = False
    event_file_ref = cast(TextIO | None, log_handler_script._event_file)
    if event_file_ref is not None:
        try:
            event_file_ref.close()
        except OSError:
            pass
        log_handler_script._event_file = None


def read_events(workdir: Path) -> list[dict[str, Any]]:
    """Read all events from the event file."""
    event_file = workdir / EVENT_FILE_NAME
    if not event_file.exists():
        return []
    events = []
    for line in event_file.read_text().strip().split("\n"):
        if line:
            events.append(json.loads(line))
    return events


class TestExtractWildcards:
    """Tests for _extract_wildcards function."""

    def test_none_wildcards(self) -> None:
        """Test with no wildcards in message."""
        assert _extract_wildcards({}) is None
        assert _extract_wildcards({"wildcards": None}) is None

    def test_dict_wildcards(self) -> None:
        """Test with dictionary wildcards."""
        msg = {"wildcards": {"sample": "A", "lane": "1"}}
        result = _extract_wildcards(msg)
        assert result == {"sample": "A", "lane": "1"}

    def test_object_wildcards(self) -> None:
        """Test with object wildcards (like Snakemake's Wildcards)."""

        class WildcardsObj:
            def __init__(self) -> None:
                self.sample = "A"
                self.lane = 1
                self._private = "hidden"

        msg = {"wildcards": WildcardsObj()}
        result = _extract_wildcards(msg)
        assert result == {"sample": "A", "lane": "1"}
        assert "_private" not in result

    def test_non_string_values_converted(self) -> None:
        """Test that non-string values are converted to strings."""
        msg = {"wildcards": {"count": 42, "flag": True}}
        result = _extract_wildcards(msg)
        assert result == {"count": "42", "flag": "True"}


class TestExtractResources:
    """Tests for _extract_resources function."""

    def test_none_resources(self) -> None:
        """Test with no resources in message."""
        assert _extract_resources({}) is None
        assert _extract_resources({"resources": None}) is None

    def test_dict_resources(self) -> None:
        """Test with dictionary resources."""
        msg = {"resources": {"mem_mb": 4000, "threads": 4, "gpu": True}}
        result = _extract_resources(msg)
        assert result == {"mem_mb": 4000, "threads": 4, "gpu": True}

    def test_object_resources(self) -> None:
        """Test with object resources."""

        class ResourcesObj:
            def __init__(self) -> None:
                self.mem_mb = 4000
                self.runtime = "2h"
                self._private = "hidden"

        msg = {"resources": ResourcesObj()}
        result = _extract_resources(msg)
        assert result == {"mem_mb": 4000, "runtime": "2h"}
        assert "_private" not in result

    def test_filters_object_repr(self) -> None:
        """Test that object representations are filtered out."""

        class SomeObject:
            pass

        msg = {"resources": {"valid": 100, "obj": SomeObject()}}
        result = _extract_resources(msg)
        assert result == {"valid": 100}

    def test_empty_resources_returns_none(self) -> None:
        """Test that empty resources dict returns None."""

        class SomeObject:
            pass

        msg = {"resources": {"obj": SomeObject()}}  # All filtered out
        result = _extract_resources(msg)
        assert result is None


class TestHandleJobInfo:
    """Tests for _handle_job_info function."""

    def test_basic_job_info(self, temp_workdir: Path) -> None:
        """Test basic job info handling."""
        msg = {
            "jobid": 1,
            "name": "align",
            "threads": 4,
            "wildcards": {"sample": "A"},
            "input": ["input.fastq"],
            "output": ["output.bam"],
        }
        # Use log_handler to ensure workflow_started is emitted
        log_handler({"level": "job_info", **msg})

        events = read_events(temp_workdir)
        # Should emit workflow_started, job_submitted, and job_started
        assert len(events) == 3
        assert events[0]["event_type"] == "workflow_started"
        assert events[1]["event_type"] == "job_submitted"
        assert events[2]["event_type"] == "job_started"

        # Check job_submitted details
        submitted = events[1]
        assert submitted["job_id"] == 1
        assert submitted["rule_name"] == "align"
        assert submitted["threads"] == 4
        assert submitted["wildcards"] == {"sample": "A"}
        assert submitted["input_files"] == ["input.fastq"]
        assert submitted["output_files"] == ["output.bam"]

    def test_stores_job_metadata(self, temp_workdir: Path) -> None:
        """Test that job metadata is stored for later correlation."""
        msg = {
            "jobid": 42,
            "rule": "process",
            "threads": 8,
            "wildcards": {"id": "test"},
        }
        _handle_job_info(msg, 2000.0)

        assert log_handler_script._job_rules[42] == "process"
        assert log_handler_script._job_threads[42] == 8
        assert log_handler_script._job_wildcards[42] == {"id": "test"}
        assert log_handler_script._job_start_times[42] == 2000.0

    def test_no_jobid(self, temp_workdir: Path) -> None:
        """Test handling when jobid is None."""
        log_handler({"level": "job_info", "name": "all"})

        events = read_events(temp_workdir)
        # Should emit workflow_started and job_submitted (no job_started without jobid)
        assert len(events) == 2
        assert events[0]["event_type"] == "workflow_started"
        assert events[1]["event_type"] == "job_submitted"
        # job_id is filtered out when None since _write_event removes None values
        assert "job_id" not in events[1]


class TestHandleJobStarted:
    """Tests for _handle_job_started function."""

    def test_job_started_with_stored_info(self, temp_workdir: Path) -> None:
        """Test job_started uses stored metadata."""
        # First store some metadata
        log_handler_script._job_rules[1] = "align"
        log_handler_script._job_wildcards[1] = {"sample": "A"}
        log_handler_script._job_threads[1] = 4

        msg = {"jobid": 1}
        _handle_job_started(msg, 1500.0)

        events = read_events(temp_workdir)
        started = events[-1]
        assert started["event_type"] == "job_started"
        assert started["job_id"] == 1
        assert started["rule_name"] == "align"
        assert started["wildcards"] == {"sample": "A"}
        assert started["threads"] == 4

    def test_job_started_no_jobid(self, temp_workdir: Path) -> None:
        """Test job_started without jobid does nothing."""
        log_handler({"level": "job_started"})  # No jobid
        events = read_events(temp_workdir)
        # Only workflow_started should be emitted (no job_started without jobid)
        assert len(events) == 1
        assert events[0]["event_type"] == "workflow_started"


class TestHandleJobFinished:
    """Tests for _handle_job_finished function."""

    def test_job_finished_with_duration(self, temp_workdir: Path) -> None:
        """Test job_finished calculates duration correctly."""
        # Store start time
        log_handler_script._job_start_times[1] = 1000.0
        log_handler_script._job_rules[1] = "align"
        log_handler_script._job_wildcards[1] = {"sample": "A"}
        log_handler_script._job_threads[1] = 4

        msg = {"jobid": 1}
        _handle_job_finished(msg, 1060.0)  # 60 seconds later

        events = read_events(temp_workdir)
        finished = events[-1]
        assert finished["event_type"] == "job_finished"
        assert finished["job_id"] == 1
        assert finished["duration"] == 60.0
        assert finished["rule_name"] == "align"

        # Metadata should be cleaned up
        assert 1 not in log_handler_script._job_start_times
        assert 1 not in log_handler_script._job_rules

    def test_job_finished_no_start_time(self, temp_workdir: Path) -> None:
        """Test job_finished without start time."""
        log_handler({"level": "job_finished", "jobid": 1})

        events = read_events(temp_workdir)
        finished = events[-1]
        assert finished["event_type"] == "job_finished"
        # duration is None and filtered out by _write_event
        assert "duration" not in finished


class TestHandleJobError:
    """Tests for _handle_job_error function."""

    def test_job_error_with_message(self, temp_workdir: Path) -> None:
        """Test job_error with error message."""
        log_handler_script._job_start_times[1] = 1000.0
        log_handler_script._job_rules[1] = "align"

        msg = {"jobid": 1, "msg": "Command failed"}
        _handle_job_error(msg, 1030.0)

        events = read_events(temp_workdir)
        error = events[-1]
        assert error["event_type"] == "job_error"
        assert error["job_id"] == 1
        assert error["error_message"] == "Command failed"
        assert error["duration"] == 30.0

    def test_job_error_uses_rule_from_message(self, temp_workdir: Path) -> None:
        """Test job_error can get rule from message."""
        msg = {"jobid": 1, "name": "process", "message": "Failed"}
        _handle_job_error(msg, 1000.0)

        events = read_events(temp_workdir)
        error = events[-1]
        assert error["rule_name"] == "process"


class TestHandleProgress:
    """Tests for _handle_progress function."""

    def test_progress_event(self, temp_workdir: Path) -> None:
        """Test progress event emission."""
        msg = {"done": 5, "total": 10}
        _handle_progress(msg, 1000.0)

        events = read_events(temp_workdir)
        progress = events[-1]
        assert progress["event_type"] == "progress"
        assert progress["completed_jobs"] == 5
        assert progress["total_jobs"] == 10


class TestLogHandler:
    """Tests for the main log_handler function."""

    def test_routes_job_info(self, temp_workdir: Path) -> None:
        """Test that job_info level routes correctly."""
        log_handler({"level": "job_info", "jobid": 1, "name": "test"})

        events = read_events(temp_workdir)
        assert any(e["event_type"] == "job_submitted" for e in events)

    def test_routes_job_finished(self, temp_workdir: Path) -> None:
        """Test that job_finished level routes correctly."""
        log_handler({"level": "job_finished", "jobid": 1})

        events = read_events(temp_workdir)
        assert any(e["event_type"] == "job_finished" for e in events)

    def test_routes_error(self, temp_workdir: Path) -> None:
        """Test that error level routes correctly."""
        log_handler({"level": "error", "jobid": 1})

        events = read_events(temp_workdir)
        assert any(e["event_type"] == "job_error" for e in events)

    def test_routes_job_error(self, temp_workdir: Path) -> None:
        """Test that job_error level routes correctly."""
        log_handler({"level": "job_error", "jobid": 1})

        events = read_events(temp_workdir)
        assert any(e["event_type"] == "job_error" for e in events)

    def test_routes_progress(self, temp_workdir: Path) -> None:
        """Test that progress level routes correctly."""
        log_handler({"level": "progress", "done": 1, "total": 5})

        events = read_events(temp_workdir)
        assert any(e["event_type"] == "progress" for e in events)

    def test_ignores_no_level(self, temp_workdir: Path) -> None:
        """Test that messages without level are ignored."""
        log_handler({"some": "data"})

        events = read_events(temp_workdir)
        # Only workflow_started should be emitted
        assert len(events) == 0

    def test_ignores_unknown_level(self, temp_workdir: Path) -> None:
        """Test that unknown levels are ignored."""
        log_handler({"level": "unknown_level"})

        events = read_events(temp_workdir)
        # Only workflow_started
        assert len(events) == 1
        assert events[0]["event_type"] == "workflow_started"


class TestWorkflowStarted:
    """Tests for workflow_started event handling."""

    def test_emitted_once(self, temp_workdir: Path) -> None:
        """Test workflow_started is only emitted once."""
        log_handler({"level": "progress", "done": 0, "total": 5})
        log_handler({"level": "progress", "done": 1, "total": 5})
        log_handler({"level": "progress", "done": 2, "total": 5})

        events = read_events(temp_workdir)
        workflow_started_count = sum(1 for e in events if e["event_type"] == "workflow_started")
        assert workflow_started_count == 1

    def test_truncates_old_events(self, temp_workdir: Path) -> None:
        """Test that workflow_started truncates existing event file."""
        # Create a pre-existing event file with old data
        event_file = temp_workdir / EVENT_FILE_NAME
        event_file.write_text('{"event_type": "old_event"}\n')

        log_handler({"level": "progress", "done": 0, "total": 5})

        events = read_events(temp_workdir)
        # Old event should be gone, only new events present
        assert not any(e["event_type"] == "old_event" for e in events)
        assert events[0]["event_type"] == "workflow_started"


class TestAcquireLockWithTimeout:
    """Tests for _acquire_lock_with_timeout function."""

    def test_acquires_lock_immediately(self, temp_workdir: Path) -> None:
        """Test that lock is acquired when available."""
        lock_file = temp_workdir / "test.lock"
        fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
        try:
            result = log_handler_script._acquire_lock_with_timeout(fd, timeout=1.0)
            assert result is True
        finally:
            os.close(fd)

    def test_timeout_when_locked(self, temp_workdir: Path) -> None:
        """Test timeout behavior when lock is held."""
        lock_file = temp_workdir / "test.lock"

        # Acquire lock in this process
        fd1 = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
        import fcntl

        fcntl.flock(fd1, fcntl.LOCK_EX)

        try:
            # Try to acquire from another fd (simulates another process)
            fd2 = os.open(str(lock_file), os.O_RDWR)
            try:
                # Use very short timeout for fast test
                result = log_handler_script._acquire_lock_with_timeout(fd2, timeout=0.05)
                assert result is False
            finally:
                os.close(fd2)
        finally:
            fcntl.flock(fd1, fcntl.LOCK_UN)
            os.close(fd1)


class TestCompleteWorkflow:
    """Integration tests for complete workflow scenarios."""

    def test_complete_job_lifecycle(self, temp_workdir: Path) -> None:
        """Test a complete job from start to finish."""
        # Job info (submission)
        log_handler(
            {
                "level": "job_info",
                "jobid": 1,
                "name": "process",
                "threads": 4,
                "wildcards": {"sample": "test"},
                "input": ["in.txt"],
                "output": ["out.txt"],
            }
        )

        # Progress update
        log_handler({"level": "progress", "done": 0, "total": 1})

        # Job finished
        log_handler({"level": "job_finished", "jobid": 1})

        # Final progress
        log_handler({"level": "progress", "done": 1, "total": 1})

        events = read_events(temp_workdir)
        event_types = [e["event_type"] for e in events]

        assert "workflow_started" in event_types
        assert "job_submitted" in event_types
        assert "job_started" in event_types
        assert "progress" in event_types
        assert "job_finished" in event_types

    def test_failed_job(self, temp_workdir: Path) -> None:
        """Test a job that fails."""
        log_handler({"level": "job_info", "jobid": 1, "name": "failing_job"})
        log_handler({"level": "job_error", "jobid": 1, "msg": "Exit code 1"})

        events = read_events(temp_workdir)
        error_events = [e for e in events if e["event_type"] == "job_error"]
        assert len(error_events) == 1
        assert error_events[0]["error_message"] == "Exit code 1"
