"""Shared test fixtures for snakesee tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def snakemake_dir(tmp_path: Path) -> Path:
    """Create a mock .snakemake directory structure."""
    smk_dir = tmp_path / ".snakemake"
    smk_dir.mkdir()
    (smk_dir / "log").mkdir()
    (smk_dir / "metadata").mkdir()
    (smk_dir / "incomplete").mkdir()
    (smk_dir / "locks").mkdir()
    return smk_dir


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    """Create a mock metadata directory with sample data."""
    import time

    meta_dir = tmp_path / ".snakemake" / "metadata"
    meta_dir.mkdir(parents=True)

    # Use recent timestamps relative to now for realistic temporal weighting tests
    now = time.time()
    day_seconds = 86400

    # Create metadata for align rule (100s duration)
    # Spread across different days for temporal weighting
    for i in range(5):
        days_ago = (4 - i) * 2  # 8, 6, 4, 2, 0 days ago (oldest to newest)
        base_time = now - (days_ago * day_seconds)
        metadata = {
            "rule": "align",
            "starttime": base_time,
            "endtime": base_time + 100.0,  # 100s duration
        }
        (meta_dir / f"align_{i}").write_text(json.dumps(metadata))

    # Create metadata for sort rule (50s duration)
    for i in range(3):
        days_ago = (2 - i) * 3  # 6, 3, 0 days ago
        base_time = now - (days_ago * day_seconds)
        metadata = {
            "rule": "sort",
            "starttime": base_time,
            "endtime": base_time + 50.0,  # 50s duration
        }
        (meta_dir / f"sort_{i}").write_text(json.dumps(metadata))

    return meta_dir


@pytest.fixture
def sample_log_content() -> str:
    """Sample snakemake log content for testing."""
    return """Building DAG of jobs...
Using shell: /bin/bash
rule align:
    jobid: 1
    output: sample1.bam
Finished job 1.
rule sort:
    jobid: 2
    output: sample1.sorted.bam
Finished job 2.
2 of 10 steps (20%) done
"""
