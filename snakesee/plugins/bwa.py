"""Plugin for BWA aligner progress parsing."""

import re

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin


class BWAPlugin(ToolProgressPlugin):
    """
    Progress parser for BWA (Burrows-Wheeler Aligner).

    Parses BWA mem/mem2 progress output which shows processed reads.

    Example BWA output::
        [M::bwa_idx_load_from_disk] read 0 ALT contigs
        [M::process] read 10000 sequences (1500000 bp)...
        [M::mem_pestat] skip orientation FF as there are not enough pairs
        [M::mem_process_seqs] Processed 10000 reads in 1.234 CPU sec
        [M::main] Real time: 2.345 sec; CPU: 1.234 sec
    """

    # Pattern: [M::mem_process_seqs] Processed 10000 reads in 1.234 CPU sec
    PROCESSED_PATTERN = re.compile(r"\[M::mem_process_seqs\]\s+Processed\s+(\d+)\s+reads")

    # Pattern: [M::process] read 10000 sequences (1500000 bp)...
    READ_PATTERN = re.compile(r"\[M::process\]\s+read\s+(\d+)\s+sequences")

    @property
    def tool_name(self) -> str:
        return "bwa"

    @property
    def tool_patterns(self) -> list[str]:
        return ["bwa mem", "bwa-mem2", "bwa aln"]

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """Check if this looks like BWA output."""
        # Check rule name
        if "bwa" in rule_name.lower():
            return True
        # Check for BWA-specific markers in log
        return "[M::bwa_" in log_content or "[M::mem_" in log_content

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """Extract progress from BWA log output."""
        total_processed = 0

        # Find all "Processed X reads" lines and sum them
        for match in self.PROCESSED_PATTERN.finditer(log_content):
            total_processed += int(match.group(1))

        # Also check for "read X sequences" pattern
        if total_processed == 0:
            for match in self.READ_PATTERN.finditer(log_content):
                total_processed += int(match.group(1))

        if total_processed > 0:
            return ToolProgress(
                items_processed=total_processed,
                items_total=None,  # BWA doesn't report total
                unit="reads",
            )

        return None
