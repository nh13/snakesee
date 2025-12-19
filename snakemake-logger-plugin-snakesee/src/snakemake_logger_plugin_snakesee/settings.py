"""Settings for snakesee logger plugin."""

from dataclasses import dataclass, field
from pathlib import Path

from snakemake_interface_logger_plugins.settings import LogHandlerSettingsBase


@dataclass
class LogHandlerSettings(LogHandlerSettingsBase):
    """Settings for the snakesee logger plugin.

    Attributes:
        event_file: Path to the event file (relative to workflow directory).
        buffer_size: Number of events to buffer before flushing.
    """

    event_file: Path = field(
        default=Path(".snakesee_events.jsonl"),
        metadata={
            "help": "Path to the event file (relative to workflow directory)",
        },
    )

    buffer_size: int = field(
        default=1,
        metadata={
            "help": "Number of events to buffer before flushing (1 = immediate)",
        },
    )
