"""Shared SQLite schema constants and helpers for tests."""

from __future__ import annotations

import sqlite3

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS snakemake_metadata (
    namespace TEXT NOT NULL,
    target TEXT NOT NULL,
    rule TEXT,
    input TEXT,
    log TEXT,
    shellcmd TEXT,
    params TEXT,
    code TEXT,
    record_format_version INTEGER,
    conda_env TEXT,
    container_img_url TEXT,
    software_stack_hash TEXT,
    job_hash INTEGER,
    starttime REAL,
    endtime REAL,
    incomplete INTEGER,
    external_jobid TEXT,
    input_checksums TEXT,
    PRIMARY KEY (namespace, target)
)
"""

CREATE_LOCKS_TABLE = """
CREATE TABLE IF NOT EXISTS snakemake_locks (
    namespace TEXT NOT NULL,
    file_path TEXT NOT NULL,
    lock_type TEXT NOT NULL,
    PRIMARY KEY (namespace, file_path, lock_type)
)
"""


def create_metadata_schema(conn: sqlite3.Connection) -> None:
    """Create the Snakemake metadata and locks tables in the given connection."""
    conn.execute(CREATE_METADATA_TABLE)
    conn.execute(CREATE_LOCKS_TABLE)
