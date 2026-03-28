"""Persistence backend abstraction for reading Snakemake metadata.

Supports both the filesystem-based backend (.snakemake/metadata/ directory)
and the newer SQLite-based backend (.snakemake/metadata.db).
"""

from snakesee.persistence.backend import IncompleteJob
from snakesee.persistence.backend import PersistenceBackend
from snakesee.persistence.backend import detect_backend
from snakesee.persistence.db import DbPersistence
from snakesee.persistence.fs import FsPersistence

__all__ = [
    "DbPersistence",
    "FsPersistence",
    "IncompleteJob",
    "PersistenceBackend",
    "detect_backend",
]
