"""Tests for the JobRegistry module."""

import pytest

from snakesee.models import JobInfo
from snakesee.state.job_registry import Job
from snakesee.state.job_registry import JobRegistry
from snakesee.state.job_registry import JobStatus


class TestJob:
    """Tests for Job dataclass."""

    def test_creation(self) -> None:
        """Test basic job creation."""
        job = Job(key="test_1", rule="align")
        assert job.key == "test_1"
        assert job.rule == "align"
        assert job.status == JobStatus.PENDING
        assert job.job_id is None

    def test_elapsed_without_start(self) -> None:
        """Test elapsed returns None without start_time."""
        job = Job(key="test_1", rule="align")
        assert job.elapsed is None

    def test_elapsed_running(self) -> None:
        """Test elapsed for running job."""
        from snakesee.state.clock import FrozenClock
        from snakesee.state.clock import reset_clock
        from snakesee.state.clock import set_clock

        clock = FrozenClock(1000.0)
        set_clock(clock)
        try:
            job = Job(key="test_1", rule="align", start_time=900.0)
            assert job.elapsed == pytest.approx(100.0)
        finally:
            reset_clock()

    def test_elapsed_completed(self) -> None:
        """Test elapsed for completed job."""
        job = Job(key="test_1", rule="align", start_time=100.0, end_time=200.0)
        assert job.elapsed == 100.0

    def test_duration_incomplete(self) -> None:
        """Test duration returns None for incomplete job."""
        job = Job(key="test_1", rule="align", start_time=100.0)
        assert job.duration is None

    def test_duration_completed(self) -> None:
        """Test duration for completed job."""
        job = Job(key="test_1", rule="align", start_time=100.0, end_time=200.0)
        assert job.duration == 100.0

    def test_to_job_info(self) -> None:
        """Test conversion to JobInfo."""
        job = Job(
            key="test_1",
            rule="align",
            job_id="job_123",
            start_time=100.0,
            end_time=200.0,
            wildcards={"sample": "A"},
            threads=4,
        )
        info = job.to_job_info()
        assert info.rule == "align"
        assert info.job_id == "job_123"
        assert info.start_time == 100.0
        assert info.end_time == 200.0
        assert info.wildcards == {"sample": "A"}
        assert info.threads == 4

    def test_from_job_info(self) -> None:
        """Test creation from JobInfo."""
        info = JobInfo(
            rule="align",
            job_id="job_123",
            start_time=100.0,
            end_time=200.0,
            wildcards={"sample": "A"},
            threads=4,
        )
        job = Job.from_job_info(info, key="test_key")
        assert job.key == "test_key"
        assert job.rule == "align"
        assert job.job_id == "job_123"
        assert job.status == JobStatus.COMPLETED
        assert job.wildcards == {"sample": "A"}

    def test_from_job_info_running(self) -> None:
        """Test creation from running JobInfo."""
        info = JobInfo(rule="align", start_time=100.0)
        job = Job.from_job_info(info)
        assert job.status == JobStatus.RUNNING


class TestJobRegistry:
    """Tests for JobRegistry."""

    def test_empty_registry(self) -> None:
        """Test empty registry state."""
        registry = JobRegistry()
        assert len(registry) == 0
        assert registry.running() == []
        assert registry.completed() == []
        assert registry.failed() == []

    def test_add_job(self) -> None:
        """Test adding a job."""
        registry = JobRegistry()
        job = Job(key="test_1", rule="align")
        registry.add(job)
        assert len(registry) == 1
        assert "test_1" in registry

    def test_get_job(self) -> None:
        """Test getting a job by key."""
        registry = JobRegistry()
        job = Job(key="test_1", rule="align")
        registry.add(job)
        assert registry.get("test_1") is job
        assert registry.get("nonexistent") is None

    def test_get_by_job_id(self) -> None:
        """Test getting a job by job_id."""
        registry = JobRegistry()
        job = Job(key="test_1", rule="align", job_id="job_123")
        registry.add(job)
        assert registry.get_by_job_id("job_123") is job
        assert registry.get_by_job_id("nonexistent") is None

    def test_get_or_create_existing(self) -> None:
        """Test get_or_create for existing job."""
        registry = JobRegistry()
        job = Job(key="test_1", rule="align")
        registry.add(job)

        result, created = registry.get_or_create("test_1", "align")
        assert result is job
        assert created is False

    def test_get_or_create_new(self) -> None:
        """Test get_or_create for new job."""
        registry = JobRegistry()
        job, created = registry.get_or_create("test_1", "align")
        assert created is True
        assert job.key == "test_1"
        assert job.rule == "align"
        assert len(registry) == 1

    def test_status_indexing(self) -> None:
        """Test jobs are indexed by status."""
        registry = JobRegistry()

        job1 = Job(key="1", rule="align", status=JobStatus.RUNNING)
        job2 = Job(key="2", rule="sort", status=JobStatus.COMPLETED)
        job3 = Job(key="3", rule="call", status=JobStatus.FAILED)

        registry.add(job1)
        registry.add(job2)
        registry.add(job3)

        assert registry.running() == [job1]
        assert registry.completed() == [job2]
        assert registry.failed() == [job3]

    def test_set_status(self) -> None:
        """Test status update with index maintenance."""
        registry = JobRegistry()
        job = Job(key="test_1", rule="align", status=JobStatus.PENDING)
        registry.add(job)

        assert registry.running() == []

        registry.set_status(job, JobStatus.RUNNING)

        assert job.status == JobStatus.RUNNING
        assert registry.running() == [job]

    def test_by_rule(self) -> None:
        """Test getting jobs by rule."""
        registry = JobRegistry()
        job1 = Job(key="1", rule="align")
        job2 = Job(key="2", rule="sort")
        job3 = Job(key="3", rule="align")

        registry.add(job1)
        registry.add(job2)
        registry.add(job3)

        align_jobs = registry.by_rule("align")
        assert len(align_jobs) == 2
        assert job1 in align_jobs
        assert job3 in align_jobs

    def test_clear(self) -> None:
        """Test clearing registry."""
        registry = JobRegistry()
        registry.add(Job(key="1", rule="align"))
        registry.add(Job(key="2", rule="sort"))

        registry.clear()

        assert len(registry) == 0
        assert registry.running() == []

    def test_apply_job_info_new(self) -> None:
        """Test applying a new JobInfo."""
        registry = JobRegistry()
        info = JobInfo(rule="align", job_id="job_123", start_time=100.0)

        job = registry.apply_job_info(info)

        assert job.rule == "align"
        assert job.job_id == "job_123"
        assert job.status == JobStatus.RUNNING
        assert len(registry) == 1

    def test_apply_job_info_update(self) -> None:
        """Test applying JobInfo updates existing job."""
        registry = JobRegistry()
        job = Job(key="job_123", rule="align", job_id="job_123")
        registry.add(job)

        info = JobInfo(
            rule="align",
            job_id="job_123",
            start_time=100.0,
            end_time=200.0,
            threads=8,
        )
        result = registry.apply_job_info(info)

        assert result is job
        assert job.start_time == 100.0
        assert job.end_time == 200.0
        assert job.threads == 8
        assert job.status == JobStatus.COMPLETED

    def test_store_threads(self) -> None:
        """Test storing threads by job_id."""
        registry = JobRegistry()
        job = Job(key="test_1", rule="align", job_id="job_123")
        registry.add(job)

        registry.store_threads("job_123", 16)

        assert job.threads == 16

    def test_all_jobs(self) -> None:
        """Test getting all jobs."""
        registry = JobRegistry()
        job1 = Job(key="1", rule="align")
        job2 = Job(key="2", rule="sort")
        registry.add(job1)
        registry.add(job2)

        all_jobs = registry.all_jobs()
        assert len(all_jobs) == 2
        assert job1 in all_jobs
        assert job2 in all_jobs


class TestJobRegistryEvents:
    """Tests for event handling in JobRegistry."""

    def test_apply_event_started(self) -> None:
        """Test applying a job started event."""
        from snakesee.events import EventType
        from snakesee.events import SnakeseeEvent

        registry = JobRegistry()
        event = SnakeseeEvent(
            event_type=EventType.JOB_STARTED,
            rule_name="align",
            job_id=123,
            timestamp=1000.0,
            threads=4,
        )

        job = registry.apply_event(event)

        assert job is not None
        assert job.rule == "align"
        assert job.status == JobStatus.RUNNING
        assert job.start_time == 1000.0
        assert job.threads == 4

    def test_apply_event_completed(self) -> None:
        """Test applying a job completed event."""
        from snakesee.events import EventType
        from snakesee.events import SnakeseeEvent

        registry = JobRegistry()
        # First start the job
        start_event = SnakeseeEvent(
            event_type=EventType.JOB_STARTED,
            rule_name="align",
            job_id=123,
            timestamp=1000.0,
        )
        job = registry.apply_event(start_event)

        # Then complete it
        complete_event = SnakeseeEvent(
            event_type=EventType.JOB_FINISHED,
            rule_name="align",
            job_id=123,
            timestamp=1100.0,
        )
        registry.apply_event(complete_event)

        assert job.status == JobStatus.COMPLETED
        assert job.end_time == 1100.0
        assert registry.completed() == [job]
        assert registry.running() == []

    def test_apply_event_failed(self) -> None:
        """Test applying a job error event."""
        from snakesee.events import EventType
        from snakesee.events import SnakeseeEvent

        registry = JobRegistry()
        start_event = SnakeseeEvent(
            event_type=EventType.JOB_STARTED,
            rule_name="align",
            job_id=123,
            timestamp=1000.0,
        )
        job = registry.apply_event(start_event)

        fail_event = SnakeseeEvent(
            event_type=EventType.JOB_ERROR,
            rule_name="align",
            job_id=123,
            timestamp=1100.0,
        )
        registry.apply_event(fail_event)

        assert job.status == JobStatus.FAILED
        assert registry.failed() == [job]


class TestJobInfoConversion:
    """Tests for JobInfo backward compatibility."""

    def test_running_job_infos(self) -> None:
        """Test getting running jobs as JobInfo."""
        registry = JobRegistry()
        job = Job(
            key="1",
            rule="align",
            status=JobStatus.RUNNING,
            start_time=100.0,
        )
        registry.add(job)

        infos = registry.running_job_infos()
        assert len(infos) == 1
        assert infos[0].rule == "align"
        assert infos[0].start_time == 100.0

    def test_completed_job_infos(self) -> None:
        """Test getting completed jobs as JobInfo."""
        registry = JobRegistry()
        job = Job(
            key="1",
            rule="align",
            status=JobStatus.COMPLETED,
            start_time=100.0,
            end_time=200.0,
        )
        registry.add(job)

        infos = registry.completed_job_infos()
        assert len(infos) == 1
        assert infos[0].duration == 100.0
