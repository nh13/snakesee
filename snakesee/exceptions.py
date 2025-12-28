"""Application-specific exceptions for snakesee.

This module provides a hierarchy of exceptions that enable more precise
error handling throughout the application. Using specific exception types
allows callers to catch and handle different error conditions appropriately.

Exception Hierarchy:
    SnakeseeError (base)
    ├── WorkflowError
    │   ├── WorkflowNotFoundError
    │   └── WorkflowParseError
    ├── ProfileError
    │   ├── ProfileNotFoundError
    │   └── InvalidProfileError
    ├── PluginError
    │   ├── PluginLoadError
    │   └── PluginExecutionError
    ├── ConfigurationError
    ├── ParsingError
    │   ├── LogParsingError
    │   └── MetadataParsingError
    ├── ValidationError
    │   ├── InvalidDurationError
    │   ├── ClockSkewError
    │   └── InvalidParameterError
    └── EventWriteError
"""

from pathlib import Path


class SnakeseeError(Exception):
    """Base exception for all snakesee errors.

    All application-specific exceptions inherit from this class,
    allowing callers to catch all snakesee errors with a single handler.
    """


class WorkflowError(SnakeseeError):
    """Base exception for workflow-related errors."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when a workflow directory or .snakemake directory is not found.

    Attributes:
        path: The path that was searched for.
        message: Human-readable error description.
    """

    def __init__(self, path: Path, message: str | None = None) -> None:
        self.path = path
        self.message = message or f"Workflow not found at {path}"
        super().__init__(self.message)


class WorkflowParseError(WorkflowError):
    """Raised when parsing workflow state fails.

    Attributes:
        path: The file or directory that could not be parsed.
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        path: Path,
        message: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.cause = cause
        self.message = message or f"Failed to parse workflow at {path}"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)


class ProfileError(SnakeseeError):
    """Base exception for profile-related errors."""


class ProfileNotFoundError(ProfileError):
    """Raised when a timing profile is not found.

    Attributes:
        path: The profile path that was not found.
        message: Human-readable error description.
    """

    def __init__(self, path: Path, message: str | None = None) -> None:
        self.path = path
        self.message = message or f"Profile not found at {path}"
        super().__init__(self.message)


class InvalidProfileError(ProfileError):
    """Raised when a timing profile is invalid or corrupted.

    Attributes:
        path: The profile path that was invalid.
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        path: Path,
        message: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.cause = cause
        self.message = message or f"Invalid profile at {path}"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)


class PluginError(SnakeseeError):
    """Base exception for plugin-related errors."""


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load.

    Attributes:
        plugin_path: Path to the plugin file or entry point name.
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        plugin_path: Path | str,
        message: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.plugin_path = plugin_path
        self.cause = cause
        self.message = message or f"Failed to load plugin from {plugin_path}"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)


class PluginExecutionError(PluginError):
    """Raised when a plugin fails during execution.

    Attributes:
        plugin_name: Name of the plugin that failed.
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        plugin_name: str,
        message: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.plugin_name = plugin_name
        self.cause = cause
        self.message = message or f"Plugin '{plugin_name}' failed during execution"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)


class ConfigurationError(SnakeseeError):
    """Raised when there is a configuration error.

    Attributes:
        parameter: The configuration parameter that is invalid.
        message: Human-readable error description.
    """

    def __init__(self, parameter: str, message: str | None = None) -> None:
        self.parameter = parameter
        self.message = message or f"Invalid configuration for '{parameter}'"
        super().__init__(self.message)


class ParsingError(SnakeseeError):
    """Base exception for parsing-related errors."""


class LogParsingError(ParsingError):
    """Raised when parsing a log file fails.

    Attributes:
        path: The log file path that could not be parsed.
        line_number: The line number where the error occurred (if applicable).
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        path: Path,
        message: str | None = None,
        line_number: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.cause = cause
        location = f" at line {line_number}" if line_number else ""
        self.message = message or f"Failed to parse log{location}: {path}"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)


class MetadataParsingError(ParsingError):
    """Raised when parsing metadata files fails.

    Attributes:
        path: The metadata file path that could not be parsed.
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        path: Path,
        message: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.cause = cause
        self.message = message or f"Failed to parse metadata: {path}"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)


class ValidationError(SnakeseeError):
    """Base exception for validation errors."""


class InvalidDurationError(ValidationError):
    """Raised when a duration value is invalid.

    Attributes:
        value: The invalid duration value.
        context: Description of where the error occurred.
        message: Human-readable error description.
    """

    def __init__(
        self,
        value: float,
        context: str | None = None,
        message: str | None = None,
    ) -> None:
        self.value = value
        self.context = context
        ctx = f" in {context}" if context else ""
        self.message = message or f"Invalid duration{ctx}: {value}"
        super().__init__(self.message)


class ClockSkewError(ValidationError):
    """Raised when clock skew is detected (negative duration).

    This typically occurs when system time has been adjusted between
    recording start_time and end_time for a job.

    Attributes:
        start_time: The recorded start time.
        end_time: The recorded end time.
        context: Description of where the error occurred.
        message: Human-readable error description.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        context: str | None = None,
        message: str | None = None,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.context = context
        ctx = f" for {context}" if context else ""
        diff = end_time - start_time
        self.message = (
            message
            or f"Clock skew detected{ctx}: end_time ({end_time:.2f}) < "
            f"start_time ({start_time:.2f}), diff={diff:.2f}s"
        )
        super().__init__(self.message)


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is invalid.

    Attributes:
        parameter: The parameter name.
        value: The invalid value.
        constraint: Description of the valid range/constraint.
        message: Human-readable error description.
    """

    def __init__(
        self,
        parameter: str,
        value: object,
        constraint: str | None = None,
        message: str | None = None,
    ) -> None:
        self.parameter = parameter
        self.value = value
        self.constraint = constraint
        constraint_msg = f" (must be {constraint})" if constraint else ""
        self.message = message or f"Invalid value for '{parameter}': {value}{constraint_msg}"
        super().__init__(self.message)


class EventWriteError(SnakeseeError):
    """Raised when writing events fails.

    Attributes:
        path: The event file path.
        message: Human-readable error description.
        cause: The underlying exception, if any.
    """

    def __init__(
        self,
        path: Path,
        message: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.cause = cause
        self.message = message or f"Failed to write event to {path}"
        if cause:
            self.message = f"{self.message}: {cause}"
        super().__init__(self.message)
