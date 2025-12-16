"""Base classes for tool-specific progress parsing plugins."""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolProgress:
    """
    Progress information extracted from a tool's log output.

    Attributes:
        items_processed: Number of items processed so far.
        items_total: Total number of items to process (None if unknown).
        unit: Unit of items (e.g., "reads", "alignments", "variants").
        percent_complete: Percentage complete (0-100), None if unknown.
        estimated_remaining_seconds: Estimated seconds remaining, None if unknown.
    """

    items_processed: int
    items_total: int | None = None
    unit: str = "items"
    percent_complete: float | None = None
    estimated_remaining_seconds: float | None = None

    def __post_init__(self) -> None:
        """Calculate percent_complete if not provided but total is known."""
        if self.percent_complete is None and self.items_total and self.items_total > 0:
            self.percent_complete = (self.items_processed / self.items_total) * 100

    @property
    def progress_str(self) -> str:
        """Human-readable progress string."""
        if self.items_total:
            return f"{self.items_processed:,}/{self.items_total:,} {self.unit}"
        return f"{self.items_processed:,} {self.unit}"

    @property
    def percent_str(self) -> str:
        """Human-readable percentage string."""
        if self.percent_complete is not None:
            return f"{self.percent_complete:.1f}%"
        return "?"


class ToolProgressPlugin(ABC):
    """
    Abstract base class for tool-specific progress parsers.

    Subclasses implement parsing logic for specific bioinformatics tools
    (e.g., BWA, STAR, samtools) to extract progress information from logs.

    Example implementation::

        class BWAPlugin(ToolProgressPlugin):
            @property
            def tool_name(self) -> str:
                return "bwa"

            @property
            def tool_patterns(self) -> list[str]:
                return ["bwa mem", "bwa-mem2"]

            def can_parse(self, rule_name: str, log_content: str) -> bool:
                return "bwa" in rule_name.lower() or "[M::mem_" in log_content

            def parse_progress(self, log_content: str) -> ToolProgress | None:
                # Parse BWA-specific progress patterns
                ...
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """
        Identifier for this tool (e.g., 'bwa', 'samtools', 'star').

        Should be lowercase and match common tool naming conventions.
        """

    @property
    def tool_patterns(self) -> list[str]:
        """
        Common command patterns that indicate this tool is being used.

        Used for initial filtering before detailed log parsing.
        Override in subclasses for tool-specific patterns.
        """
        return [self.tool_name]

    @abstractmethod
    def can_parse(self, rule_name: str, log_content: str) -> bool:
        """
        Determine if this plugin can parse the given log content.

        Args:
            rule_name: Name of the Snakemake rule.
            log_content: Content of the rule's log file.

        Returns:
            True if this plugin can extract progress from this log.
        """

    @abstractmethod
    def parse_progress(self, log_content: str) -> ToolProgress | None:
        """
        Extract progress information from log content.

        Args:
            log_content: Content of the rule's log file.

        Returns:
            ToolProgress if progress could be extracted, None otherwise.
        """

    def parse_progress_from_file(self, log_path: Path) -> ToolProgress | None:
        """
        Extract progress from a log file.

        Args:
            log_path: Path to the log file.

        Returns:
            ToolProgress if progress could be extracted, None otherwise.
        """
        try:
            content = log_path.read_text(errors="ignore")
            return self.parse_progress(content)
        except OSError:
            return None
