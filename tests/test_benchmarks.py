"""Performance benchmarks for snakesee (TC-L3).

Run benchmarks with: pytest tests/test_benchmarks.py --benchmark-only
Compare results: pytest tests/test_benchmarks.py --benchmark-compare

These tests are skipped by default in normal test runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

    from snakesee.estimation import TimeEstimator

# Skip benchmarks by default (run with --benchmark-only or --benchmark-enable)
pytestmark = pytest.mark.benchmark


class TestParserBenchmarks:
    """Benchmarks for parser performance on large datasets."""

    @pytest.fixture
    def large_log_file(self, tmp_path: Path) -> Path:
        """Generate a large synthetic log file."""
        log_file = tmp_path / "large.log"
        lines = []

        # Generate 1000 jobs
        for i in range(1000):
            rule = f"rule_{i % 10}"
            lines.extend(
                [
                    f"rule {rule}:",
                    f"    jobid: {i}",
                    f"    wildcards: sample=sample{i}, batch={i % 5}",
                    f"    threads: {2 + i % 6}",
                    "[Mon Dec 16 10:00:00 2024]",
                    f"Finished job {i}.",
                    f"{i + 1} of 1000 steps ({(i + 1) / 10:.1f}%) done",
                ]
            )

        log_file.write_text("\n".join(lines))
        return log_file

    @pytest.fixture
    def large_metadata_dir(self, tmp_path: Path) -> Path:
        """Generate a large metadata directory."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        base_time = time.time() - 86400  # Yesterday

        for i in range(500):
            rule = f"rule_{i % 10}"
            duration = 50 + (i % 100)
            metadata = {
                "rule": rule,
                "starttime": base_time + i * 100,
                "endtime": base_time + i * 100 + duration,
                "wildcards": {"sample": f"sample{i}", "batch": str(i % 5)},
            }
            (metadata_dir / f"output_{i}").write_text(json.dumps(metadata))

        return metadata_dir

    def test_benchmark_log_parsing(self, benchmark: BenchmarkFixture, large_log_file: Path) -> None:
        """Benchmark incremental log parsing."""
        from snakesee.parser import IncrementalLogReader

        def parse_log() -> int:
            reader = IncrementalLogReader(large_log_file)
            return reader.read_new_lines()

        result = benchmark(parse_log)
        assert result > 0

    def test_benchmark_metadata_parsing(
        self, benchmark: BenchmarkFixture, large_metadata_dir: Path
    ) -> None:
        """Benchmark metadata file parsing."""
        from snakesee.parser import parse_metadata_files

        def parse_metadata() -> int:
            return len(list(parse_metadata_files(large_metadata_dir)))

        result = benchmark(parse_metadata)
        assert result == 500

    def test_benchmark_wildcard_parsing(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark wildcard string parsing."""
        from snakesee.parser import _parse_wildcards

        # Complex wildcard string
        wildcard_str = ", ".join([f"key{i}=value{i}" for i in range(20)])

        def parse_wildcards() -> dict[str, str] | None:
            return _parse_wildcards(wildcard_str)

        result = benchmark(parse_wildcards)
        assert result is not None
        assert len(result) == 20


class TestEstimatorBenchmarks:
    """Benchmarks for estimator performance."""

    @pytest.fixture
    def populated_estimator(self, tmp_path: Path) -> TimeEstimator:
        """Create an estimator with historical data."""
        from snakesee.estimation import TimeEstimator

        # Create metadata for training
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        base_time = time.time() - 86400

        for i in range(200):
            rule = f"rule_{i % 10}"
            duration = 50 + (i % 100)
            metadata = {
                "rule": rule,
                "starttime": base_time + i * 100,
                "endtime": base_time + i * 100 + duration,
            }
            (metadata_dir / f"output_{i}").write_text(json.dumps(metadata))

        estimator = TimeEstimator()
        estimator.load_from_metadata(metadata_dir)
        return estimator

    def test_benchmark_estimate_remaining(
        self, benchmark: BenchmarkFixture, populated_estimator: TimeEstimator
    ) -> None:
        """Benchmark remaining time estimation."""
        from snakesee.models import WorkflowProgress
        from snakesee.models import WorkflowStatus

        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=100,
            completed_jobs=50,
            failed_jobs=0,
            running_jobs=[],
            recent_completions=[],
            failed_jobs_list=[],
            incomplete_jobs_list=[],
            start_time=time.time() - 600,
        )

        def estimate() -> Any:
            result = populated_estimator.estimate_remaining(progress)
            return result.seconds_remaining

        result = benchmark(estimate)
        assert result >= 0

    def test_benchmark_get_rule_estimate(
        self, benchmark: BenchmarkFixture, populated_estimator: TimeEstimator
    ) -> None:
        """Benchmark per-rule estimation."""

        def get_estimate() -> Any:
            duration, variance = populated_estimator.get_estimate_for_job(
                rule="rule_0",
                threads=4,
                wildcards={"sample": "test"},
                input_size=None,
            )
            return duration

        result = benchmark(get_estimate)
        assert result > 0


class TestValidationBenchmarks:
    """Benchmarks for validation performance."""

    def test_benchmark_event_accumulation(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark event accumulation for large workflows."""
        from snakesee.events import EventType
        from snakesee.events import SnakeseeEvent
        from snakesee.validation import EventAccumulator

        def accumulate_events() -> int:
            acc = EventAccumulator()

            # Simulate 500 jobs
            base_time = time.time()
            for i in range(500):
                rule = f"rule_{i % 10}"
                # Submit event
                acc.process_event(
                    SnakeseeEvent(
                        event_type=EventType.JOB_SUBMITTED,
                        timestamp=base_time + i,
                        rule_name=rule,
                        job_id=i,
                        threads=4,
                    )
                )
                # Started event
                acc.process_event(
                    SnakeseeEvent(
                        event_type=EventType.JOB_STARTED,
                        timestamp=base_time + i + 0.1,
                        rule_name=rule,
                        job_id=i,
                    )
                )
                # Finished event
                acc.process_event(
                    SnakeseeEvent(
                        event_type=EventType.JOB_FINISHED,
                        timestamp=base_time + i + 50 + (i % 100),
                        rule_name=rule,
                        job_id=i,
                        duration=50.0 + i % 100,
                    )
                )

            return len(acc.jobs)

        result = benchmark(accumulate_events)
        assert result == 500


class TestTUIBenchmarks:
    """Benchmarks for TUI rendering performance."""

    def test_benchmark_format_duration(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark duration formatting."""
        from snakesee.formatting import format_duration

        def format_many() -> list[str]:
            return [format_duration(i * 60.5) for i in range(100)]

        result = benchmark(format_many)
        assert len(result) == 100

    def test_benchmark_format_eta(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark ETA formatting."""
        from snakesee.formatting import format_eta

        def format_many() -> list[str]:
            return [format_eta(i * 60.5, confidence=0.8) for i in range(100)]

        result = benchmark(format_many)
        assert len(result) == 100
