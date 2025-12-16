"""Plugin for STAR aligner progress parsing."""

import re

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin


class STARPlugin(ToolProgressPlugin):
    """
    Progress parser for STAR (Spliced Transcripts Alignment to a Reference).

    Parses STAR progress output which shows alignment progress.

    Example STAR output::
        STAR version=2.7.10a
        ...
        Finished 10000000 paired reads
        Finished 20000000 paired reads
        ...
        Uniquely mapped reads % |	85.00%
    """

    # Pattern: Finished 10000000 paired reads
    FINISHED_PATTERN = re.compile(r"Finished\s+(\d+)\s+(?:paired\s+)?reads")

    # Pattern from Log.progress.out: 10000000 reads processed
    PROGRESS_PATTERN = re.compile(r"(\d+)\s+reads?\s+processed")

    # Percentage from Log.final.out
    MAPPED_PATTERN = re.compile(r"Uniquely mapped reads %\s*\|\s*([\d.]+)%")

    @property
    def tool_name(self) -> str:
        return "star"

    @property
    def tool_patterns(self) -> list[str]:
        return ["STAR", "star"]

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """Check if this looks like STAR output."""
        if "star" in rule_name.lower():
            return True
        return "STAR version" in log_content or "STAR --" in log_content

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """Extract progress from STAR log output."""
        # Find all "Finished X reads" lines
        finished_matches = list(self.FINISHED_PATTERN.finditer(log_content))
        if finished_matches:
            last_count = int(finished_matches[-1].group(1))
            return ToolProgress(
                items_processed=last_count,
                items_total=None,
                unit="reads",
            )

        # Check for progress file format
        progress_matches = list(self.PROGRESS_PATTERN.finditer(log_content))
        if progress_matches:
            last_count = int(progress_matches[-1].group(1))
            return ToolProgress(
                items_processed=last_count,
                items_total=None,
                unit="reads",
            )

        return None
