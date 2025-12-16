"""Plugin for samtools progress parsing."""

import re

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin


class SamtoolsSortPlugin(ToolProgressPlugin):
    """
    Progress parser for samtools sort.

    Parses samtools sort progress output which shows processed reads.

    Example samtools sort output::
        [bam_sort_core] merging from 4 files and 1 in-memory blocks...
        [bam_sort_core] read 1000000 records...
    """

    # Pattern: read 1000000 records
    RECORDS_PATTERN = re.compile(r"read\s+(\d+)\s+records")

    @property
    def tool_name(self) -> str:
        return "samtools-sort"

    @property
    def tool_patterns(self) -> list[str]:
        return ["samtools sort", "samtools view"]

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """Check if this looks like samtools sort output."""
        if "sort" in rule_name.lower() and "sam" in rule_name.lower():
            return True
        return "[bam_sort_core]" in log_content

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """Extract progress from samtools sort log output."""
        total_records = 0

        for match in self.RECORDS_PATTERN.finditer(log_content):
            records = int(match.group(1))
            if records > total_records:
                total_records = records

        if total_records > 0:
            return ToolProgress(
                items_processed=total_records,
                items_total=None,
                unit="records",
            )

        return None


class SamtoolsIndexPlugin(ToolProgressPlugin):
    """
    Progress parser for samtools index.

    samtools index doesn't provide progress, but we can detect completion.
    """

    @property
    def tool_name(self) -> str:
        return "samtools-index"

    @property
    def tool_patterns(self) -> list[str]:
        return ["samtools index"]

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """Check if this looks like samtools index."""
        return "index" in rule_name.lower() and (
            "sam" in rule_name.lower() or "bam" in rule_name.lower()
        )

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """samtools index doesn't report progress."""
        return None
