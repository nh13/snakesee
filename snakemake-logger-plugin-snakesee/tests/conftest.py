"""Pytest configuration for snakemake-logger-plugin-snakesee tests.

TODO: Once https://github.com/snakemake/snakemake/pull/3897 is merged and
released, we can use SnakemakeApi.get_log_handlers() to inspect logger state
(e.g., RUN_INFO events) after workflow execution, enabling better integration
testing of the logger plugin.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
