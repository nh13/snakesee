"""Plugin for fgbio tool progress parsing."""

import re

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin


class FgbioPlugin(ToolProgressPlugin):
    """
    Progress parser for fgbio tools.

    fgbio is a toolkit for working with genomic and high throughput sequencing data.
    It uses HTSJDK's ProgressLogger which outputs messages like:

        [INFO] Processed 1,000,000 records. Elapsed time: 00:01:30s.
        [progress] Read 5000000 records from BAM file.
        [INFO] Grouped 1234567 records into 123456 read pairs.

    This plugin detects fgbio by rule name or log content patterns.
    """

    # Pattern: "Processed 1,000,000 records" (HTSJDK ProgressLogger format)
    PROCESSED_PATTERN = re.compile(
        r"[Pp]rocessed\s+([\d,]+)\s+(\w+)",
        re.IGNORECASE,
    )

    # Pattern: "Read 5000000 records from BAM"
    READ_PATTERN = re.compile(
        r"[Rr]ead\s+([\d,]+)\s+(\w+)",
        re.IGNORECASE,
    )

    # Pattern: "Grouped 1234567 records into X"
    GROUPED_PATTERN = re.compile(
        r"[Gg]rouped\s+([\d,]+)\s+(\w+)",
        re.IGNORECASE,
    )

    # Pattern: "Wrote 1000000 records"
    WROTE_PATTERN = re.compile(
        r"[Ww]rote\s+([\d,]+)\s+(\w+)",
        re.IGNORECASE,
    )

    # Pattern: "Finished. Processed X records in Y seconds."
    FINISHED_PATTERN = re.compile(
        r"[Ff]inished.*?[Pp]rocessed\s+([\d,]+)\s+(\w+)",
        re.IGNORECASE,
    )

    @property
    def tool_name(self) -> str:
        return "fgbio"

    @property
    def tool_patterns(self) -> list[str]:
        return ["fgbio", "CallMolecularConsensusReads", "GroupReadsByUmi", "FilterConsensusReads"]

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """Check if this looks like fgbio output."""
        rule_lower = rule_name.lower()
        if "fgbio" in rule_lower:
            return True
        # Common fgbio tool names in rules
        fgbio_tools = [
            "callmolecularconsensusreads",
            "groupreadsbyumi",
            "filterconsensusreads",
            "callduplexxxconsensusreads",
            "annotatebamwithumi",
            "correctumis",
            "extractumisfrombam",
            "demuxfastqs",
            "trimfastq",
        ]
        if any(tool in rule_lower for tool in fgbio_tools):
            return True
        # Check log content for fgbio signatures
        if "com.fulcrumgenomics" in log_content:
            return True
        if "fgbio" in log_content.lower() and "Processed" in log_content:
            return True
        return False

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """Extract progress from fgbio log output."""
        # Try different patterns in order of preference
        patterns = [
            (self.FINISHED_PATTERN, "finished"),
            (self.PROCESSED_PATTERN, "processed"),
            (self.READ_PATTERN, "read"),
            (self.GROUPED_PATTERN, "grouped"),
            (self.WROTE_PATTERN, "wrote"),
        ]

        best_count = 0
        best_unit = "records"

        for pattern, _ in patterns:
            matches = list(pattern.finditer(log_content))
            if matches:
                # Get the last (most recent) match
                last_match = matches[-1]
                count_str = last_match.group(1).replace(",", "")
                count = int(count_str)
                unit = last_match.group(2).lower()

                # Normalize unit names
                if unit in ("record", "records"):
                    unit = "records"
                elif unit in ("read", "reads"):
                    unit = "reads"
                elif unit in ("pair", "pairs"):
                    unit = "read pairs"

                if count > best_count:
                    best_count = count
                    best_unit = unit

        if best_count > 0:
            return ToolProgress(
                items_processed=best_count,
                items_total=None,
                unit=best_unit,
            )

        return None
