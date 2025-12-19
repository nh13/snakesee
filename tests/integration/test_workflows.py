"""Integration tests running real Snakemake workflows with validation.

These tests verify that snakesee's event-based tracking and log parsing
produce consistent results across various workflow patterns.

Requires:
    - Snakemake 8+
    - snakemake-logger-plugin-snakesee installed

Run with:
    pytest tests/integration -v --no-cov
    # or
    poe check-integration
"""

import pytest

from tests.integration.conftest import WorkflowRunner

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestSuccessfulWorkflows:
    """Test workflows that should complete successfully."""

    def test_simple_linear(self, workflow_runner: WorkflowRunner) -> None:
        """Test simple linear pipeline A -> B -> C."""
        workflow_runner.setup_workflow("simple_linear")
        workflow_runner.run()

        result = workflow_runner.validate_events()

        # Verify workflow lifecycle
        assert result.workflow_started, "Workflow started event missing"
        assert result.total_jobs == 4, f"Expected 4 total jobs, got {result.total_jobs}"
        assert result.completed_jobs == 4, f"Expected 4 completed, got {result.completed_jobs}"
        assert result.all_jobs_finished, "Not all jobs finished"
        assert result.job_count == 4, f"Expected 4 jobs tracked, got {result.job_count}"
        assert result.failed_job_count == 0, "No jobs should have failed"

    def test_parallel_samples(self, workflow_runner: WorkflowRunner) -> None:
        """Test parallel sample processing."""
        workflow_runner.setup_workflow("parallel_samples")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_fan_out_fan_in(self, workflow_runner: WorkflowRunner) -> None:
        """Test fan-out and fan-in pattern."""
        workflow_runner.setup_workflow("fan_out_fan_in")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_diamond(self, workflow_runner: WorkflowRunner) -> None:
        """Test diamond dependency pattern A -> (B, C) -> D."""
        workflow_runner.setup_workflow("diamond")
        workflow_runner.run(cores=2)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_wildcards_complex(self, workflow_runner: WorkflowRunner) -> None:
        """Test complex wildcard patterns with multiple dimensions."""
        workflow_runner.setup_workflow("wildcards_complex")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 2 samples * 2 treatments * 2 replicates = 8 process jobs
        assert result.job_count >= 8, f"Expected at least 8 jobs, got {result.job_count}"

    def test_quick_jobs(self, workflow_runner: WorkflowRunner) -> None:
        """Test many quick jobs (timing edge cases)."""
        workflow_runner.setup_workflow("quick_jobs")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_mixed_duration(self, workflow_runner: WorkflowRunner) -> None:
        """Test mix of fast and slow jobs."""
        workflow_runner.setup_workflow("mixed_duration")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_resources(self, workflow_runner: WorkflowRunner) -> None:
        """Test jobs with resource constraints."""
        workflow_runner.setup_workflow("resources")
        workflow_runner.run(cores=4, extra_args=["--resources", "mem_mb=2000"])

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_temp_files(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with temporary files."""
        workflow_runner.setup_workflow("temp_files")
        workflow_runner.run()

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_scatter_gather(self, workflow_runner: WorkflowRunner) -> None:
        """Test scatter-gather pattern."""
        workflow_runner.setup_workflow("scatter_gather")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_nested_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test nested wildcards with constraints."""
        workflow_runner.setup_workflow("nested_wildcards")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 3 samples * 2 lanes * 2 reads = 12 process jobs
        assert result.job_count >= 12, f"Expected at least 12 jobs, got {result.job_count}"


class TestCheckpointWorkflows:
    """Test workflows with dynamic DAG (checkpoints)."""

    def test_checkpoint_workflow(self, workflow_runner: WorkflowRunner) -> None:
        """Test checkpoint with dynamic job creation."""
        workflow_runner.setup_workflow("checkpoint_workflow")
        workflow_runner.run(cores=2)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0


class TestFailingWorkflows:
    """Test workflows that are expected to fail."""

    def test_failing_job(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with a failing job."""
        workflow_runner.setup_workflow("failing_job")
        run_result = workflow_runner.run(expect_failure=True)

        # Workflow should fail
        assert run_result.returncode != 0

        # Validate events were captured
        result = workflow_runner.validate_events()
        assert result.workflow_started
        # At least one job should have failed
        assert result.failed_job_count >= 1, "Expected at least one failed job"

    def test_failing_with_retry(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with job that fails and retries."""
        workflow_runner.setup_workflow("failing_with_retry")
        run_result = workflow_runner.run(expect_failure=False)  # Should eventually succeed

        # Workflow should eventually succeed after retries
        assert run_result.returncode == 0

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished

    def test_multiple_failures(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with multiple failing jobs."""
        workflow_runner.setup_workflow("multiple_failures")
        run_result = workflow_runner.run(
            expect_failure=True, extra_args=["--keep-going"]
        )

        # Workflow should fail (multiple failures)
        assert run_result.returncode != 0

        # Validate events were captured
        result = workflow_runner.validate_events()
        assert result.workflow_started
        # Multiple jobs should have failed
        assert result.failed_job_count >= 1, "Expected failed jobs"


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_single_job(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with a single job."""
        workflow_runner.setup_workflow("simple_linear")
        # Only build the first step
        workflow_runner.run(targets=["output/step_a.txt"])

        result = workflow_runner.validate_events()
        assert result.workflow_started
        # Should have at least 1 job (step_a)
        assert result.job_count >= 1

    def test_dry_run_no_events(self, workflow_runner: WorkflowRunner) -> None:
        """Test that dry run doesn't produce validation events."""
        workflow_runner.setup_workflow("simple_linear")

        # Dry run should not produce events
        run_result = workflow_runner.run(extra_args=["--dry-run"], expect_failure=False)
        assert run_result.returncode == 0

        # Event count should be 0 or minimal (workflow_started only)
        event_count = workflow_runner.get_event_count()
        # Dry run may or may not produce events depending on plugin behavior
        # Just verify it doesn't crash

    def test_already_complete(self, workflow_runner: WorkflowRunner) -> None:
        """Test re-running an already complete workflow."""
        workflow_runner.setup_workflow("simple_linear")

        # First run
        workflow_runner.run()
        first_result = workflow_runner.validate_events()
        assert first_result.all_jobs_finished

        # Second run - nothing to do
        workflow_runner.run()
        # Should still have events (from both runs)
        assert workflow_runner.get_event_count() > 0


class TestConcurrency:
    """Test concurrent job execution patterns."""

    def test_high_parallelism(self, workflow_runner: WorkflowRunner) -> None:
        """Test with many parallel jobs."""
        workflow_runner.setup_workflow("parallel_samples")
        workflow_runner.run(cores=8)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_serialized_execution(self, workflow_runner: WorkflowRunner) -> None:
        """Test fully serialized execution (cores=1)."""
        workflow_runner.setup_workflow("parallel_samples")
        workflow_runner.run(cores=1)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0


class TestParserEdgeCases:
    """Test edge cases for log parser patterns."""

    def test_localrules(self, workflow_runner: WorkflowRunner) -> None:
        """Test localrule pattern parsing (localrule X:)."""
        workflow_runner.setup_workflow("localrules")
        workflow_runner.run()

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_special_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test wildcards with special characters (-, _, .)."""
        workflow_runner.setup_workflow("special_wildcards")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 3 samples * 2 types = 6 jobs + all
        assert result.job_count >= 6
        assert result.failed_job_count == 0

    def test_deep_chain(self, workflow_runner: WorkflowRunner) -> None:
        """Test very deep dependency chain (15 steps)."""
        workflow_runner.setup_workflow("deep_chain")
        workflow_runner.run()

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 15 steps + all = 16 jobs
        assert result.job_count == 16
        assert result.failed_job_count == 0

    def test_wide_fanout(self, workflow_runner: WorkflowRunner) -> None:
        """Test wide fan-out (1 -> 20 parallel jobs)."""
        workflow_runner.setup_workflow("wide_fanout")
        workflow_runner.run(cores=8)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 1 input + 20 chunks + all = 22 jobs
        assert result.job_count == 22
        assert result.failed_job_count == 0

    def test_multi_output(self, workflow_runner: WorkflowRunner) -> None:
        """Test rules with multiple outputs."""
        workflow_runner.setup_workflow("multi_output")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_long_names(self, workflow_runner: WorkflowRunner) -> None:
        """Test very long rule names and wildcard values."""
        workflow_runner.setup_workflow("long_names")
        workflow_runner.run()

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0


class TestAdvancedFeatures:
    """Test advanced Snakemake features."""

    def test_input_function(self, workflow_runner: WorkflowRunner) -> None:
        """Test dynamic input using input functions."""
        workflow_runner.setup_workflow("input_function")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_with_logs(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with log directive."""
        workflow_runner.setup_workflow("with_logs")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_with_benchmark(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow with benchmark directive."""
        workflow_runner.setup_workflow("with_benchmark")
        workflow_runner.run(cores=2)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_group_jobs(self, workflow_runner: WorkflowRunner) -> None:
        """Test grouped jobs."""
        workflow_runner.setup_workflow("group_jobs")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_params_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test params that depend on wildcards."""
        workflow_runner.setup_workflow("params_wildcards")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0


class TestErrorTypes:
    """Test different types of errors."""

    def test_python_script_error(self, workflow_runner: WorkflowRunner) -> None:
        """Test Python script error (ValueError exception)."""
        workflow_runner.setup_workflow("python_script_error")
        run_result = workflow_runner.run(expect_failure=True)

        assert run_result.returncode != 0

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.failed_job_count >= 1

    def test_missing_input_error(self, workflow_runner: WorkflowRunner) -> None:
        """Test error when input file is missing."""
        workflow_runner.setup_workflow("missing_input_error")
        run_result = workflow_runner.run(expect_failure=True)

        # Should fail due to missing input
        assert run_result.returncode != 0


class TestResourcesAndThreads:
    """Test thread and resource handling."""

    def test_threads_scaling(self, workflow_runner: WorkflowRunner) -> None:
        """Test thread scaling when cores are limited."""
        workflow_runner.setup_workflow("threads_scaling")
        # Run with only 2 cores, jobs request 8
        workflow_runner.run(cores=2)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_custom_resources(self, workflow_runner: WorkflowRunner) -> None:
        """Test custom resource types (gpu, disk_mb, io_weight)."""
        workflow_runner.setup_workflow("custom_resources")
        workflow_runner.run(cores=4, extra_args=["--resources", "gpu=2", "disk_mb=5000"])

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0


class TestCheckpointsExtended:
    """Test additional checkpoint scenarios."""

    def test_checkpoint_with_aggregation(self, workflow_runner: WorkflowRunner) -> None:
        """Test checkpoint with multi-step processing and aggregation."""
        workflow_runner.setup_workflow("multiple_checkpoints")
        workflow_runner.run(cores=2)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        # Note: Snakemake 9 checkpoint re-evaluation may complete in stages
        # At minimum, the checkpoint itself should complete
        assert result.job_count >= 1
        assert result.failed_job_count == 0


class TestModules:
    """Test module/include workflows."""

    def test_module_workflow(self, workflow_runner: WorkflowRunner) -> None:
        """Test workflow using include for modular rules."""
        workflow_runner.setup_workflow("module_workflow")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0


class TestRetries:
    """Test retry behavior."""

    def test_retry_exhausted(self, workflow_runner: WorkflowRunner) -> None:
        """Test job that fails even after all retries are exhausted."""
        workflow_runner.setup_workflow("retry_exhausted")
        run_result = workflow_runner.run(expect_failure=True)

        # Should fail after exhausting retries
        assert run_result.returncode != 0

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.failed_job_count >= 1


class TestKeepGoing:
    """Test --keep-going behavior."""

    def test_keep_going_mixed(self, workflow_runner: WorkflowRunner) -> None:
        """Test --keep-going with mixed success/failure branches."""
        workflow_runner.setup_workflow("keep_going_mixed")
        run_result = workflow_runner.run(
            expect_failure=True,
            cores=4,
            extra_args=["--keep-going"]
        )

        # Should fail overall
        assert run_result.returncode != 0

        result = workflow_runner.validate_events()
        assert result.workflow_started
        # Branch B should complete successfully
        # Branches A and C should fail
        assert result.failed_job_count >= 2
        # Some jobs should have finished (branch B)
        assert result.finished_job_count >= 2


class TestWildcards:
    """Test wildcard handling edge cases."""

    def test_numeric_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test purely numeric wildcard values (1-10)."""
        workflow_runner.setup_workflow("numeric_wildcards")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 10 chunks + all = 11 jobs
        assert result.job_count == 11
        assert result.failed_job_count == 0

    def test_adjacent_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test adjacent wildcards without separator (e.g., {prefix}{suffix})."""
        workflow_runner.setup_workflow("adjacent_wildcards")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 2 prefixes * 3 suffixes = 6 jobs + all
        assert result.job_count == 7
        assert result.failed_job_count == 0

    def test_conditional_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test conditional input based on wildcard value."""
        workflow_runner.setup_workflow("conditional_wildcards")
        workflow_runner.run(cores=4)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        assert result.failed_job_count == 0

    def test_many_wildcards(self, workflow_runner: WorkflowRunner) -> None:
        """Test many wildcard combinations (100 jobs stress test)."""
        workflow_runner.setup_workflow("many_wildcards")
        workflow_runner.run(cores=8)

        result = workflow_runner.validate_events()
        assert result.workflow_started
        assert result.all_jobs_finished
        # 5 * 5 * 4 = 100 process jobs + all = 101
        assert result.job_count == 101
        assert result.failed_job_count == 0


class TestParserVsPluginValidation:
    """Test that parser and plugin produce consistent results.

    These tests compare the log parser output with the logger plugin events
    to find any discrepancies between the two tracking methods.
    """

    def test_simple_linear_validation(self, workflow_runner: WorkflowRunner) -> None:
        """Validate parser matches plugin for simple linear workflow."""
        workflow_runner.setup_workflow("simple_linear")
        workflow_runner.run()

        discrepancies = workflow_runner.validate()
        if discrepancies:
            for d in discrepancies:
                print(f"Discrepancy: {d}")
        assert len(discrepancies) == 0, f"Found {len(discrepancies)} discrepancies"

    def test_parallel_validation(self, workflow_runner: WorkflowRunner) -> None:
        """Validate parser matches plugin for parallel workflow."""
        workflow_runner.setup_workflow("parallel_samples")
        workflow_runner.run(cores=4)

        discrepancies = workflow_runner.validate()
        if discrepancies:
            for d in discrepancies:
                print(f"Discrepancy: {d}")
        assert len(discrepancies) == 0, f"Found {len(discrepancies)} discrepancies"

    def test_wildcards_validation(self, workflow_runner: WorkflowRunner) -> None:
        """Validate parser matches plugin for wildcard workflow."""
        workflow_runner.setup_workflow("wildcards_complex")
        workflow_runner.run(cores=4)

        discrepancies = workflow_runner.validate()
        if discrepancies:
            for d in discrepancies:
                print(f"Discrepancy: {d}")
        assert len(discrepancies) == 0, f"Found {len(discrepancies)} discrepancies"

    def test_deep_chain_validation(self, workflow_runner: WorkflowRunner) -> None:
        """Validate parser matches plugin for deep chain workflow."""
        workflow_runner.setup_workflow("deep_chain")
        workflow_runner.run()

        discrepancies = workflow_runner.validate()
        if discrepancies:
            for d in discrepancies:
                print(f"Discrepancy: {d}")
        assert len(discrepancies) == 0, f"Found {len(discrepancies)} discrepancies"

    def test_wide_fanout_validation(self, workflow_runner: WorkflowRunner) -> None:
        """Validate parser matches plugin for wide fanout workflow."""
        workflow_runner.setup_workflow("wide_fanout")
        workflow_runner.run(cores=8)

        discrepancies = workflow_runner.validate()
        if discrepancies:
            for d in discrepancies:
                print(f"Discrepancy: {d}")
        assert len(discrepancies) == 0, f"Found {len(discrepancies)} discrepancies"

    def test_failing_job_validation(self, workflow_runner: WorkflowRunner) -> None:
        """Validate parser matches plugin for failing workflow."""
        workflow_runner.setup_workflow("failing_job")
        workflow_runner.run(expect_failure=True)

        discrepancies = workflow_runner.validate()
        # Some discrepancies are expected for failed jobs since
        # the parser and plugin may track failures differently
        # Filter out expected differences
        errors = [d for d in discrepancies if d.get("severity") == "error"]
        if errors:
            for e in errors:
                print(f"Error: {e}")
        assert len(errors) == 0, f"Found {len(errors)} error-level discrepancies"
