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
    └── ConfigurationError
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
