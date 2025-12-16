"""Plugin for fastp QC tool progress parsing."""

import re

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin


class FastpPlugin(ToolProgressPlugin):
    """
    Progress parser for fastp (FASTQ preprocessor).

    Parses fastp progress output which shows processed reads.

    Example fastp output::
        Read1 before filtering:
        total reads: 10000000
        total bases: 1500000000
        ...
        Filtering result:
        reads passed filter: 9800000
        reads failed due to low quality: 100000
    """

    # Pattern: total reads: 10000000
    TOTAL_READS_PATTERN = re.compile(r"total reads:\s+(\d+)")

    # Pattern: reads passed filter: 9800000
    PASSED_PATTERN = re.compile(r"reads passed filter:\s+(\d+)")

    # Progress pattern: Processing 50.00% of reads
    PROGRESS_PATTERN = re.compile(r"Processing\s+([\d.]+)%")

    @property
    def tool_name(self) -> str:
        return "fastp"

    @property
    def tool_patterns(self) -> list[str]:
        return ["fastp"]

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """Check if this looks like fastp output."""
        if "fastp" in rule_name.lower():
            return True
        return "fastp" in log_content.lower() or "Read1 before filtering" in log_content

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """Extract progress from fastp log output."""
        # Check for explicit progress percentage
        progress_matches = list(self.PROGRESS_PATTERN.finditer(log_content))
        if progress_matches:
            last_progress = float(progress_matches[-1].group(1))
            return ToolProgress(
                items_processed=int(last_progress),
                items_total=100,
                unit="%",
                percent_complete=last_progress,
            )

        # Look for total reads and passed reads
        total_match = self.TOTAL_READS_PATTERN.search(log_content)
        passed_match = self.PASSED_PATTERN.search(log_content)

        if total_match and passed_match:
            total = int(total_match.group(1))
            passed = int(passed_match.group(1))
            return ToolProgress(
                items_processed=passed,
                items_total=total,
                unit="reads",
            )

        if total_match:
            total = int(total_match.group(1))
            return ToolProgress(
                items_processed=total,
                items_total=None,
                unit="reads",
            )

        return None
