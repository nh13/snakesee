"""File-based plugin loading with security checks.

This module handles loading plugins from Python files in user directories.
It includes security checks for symlinks and world-writable directories.
"""

import importlib.util
import logging
import stat
import sys
from pathlib import Path

from snakesee.plugins.base import PLUGIN_API_VERSION
from snakesee.plugins.base import PluginMetadata
from snakesee.plugins.base import ToolProgressPlugin

logger = logging.getLogger(__name__)

# User plugin directories (searched in order)
USER_PLUGIN_DIRS: list[Path] = [
    Path.home() / ".snakesee" / "plugins",
    Path.home() / ".config" / "snakesee" / "plugins",
]

# Cache for loaded user plugins
_user_plugins: list[ToolProgressPlugin] | None = None


def validate_plugin(plugin: ToolProgressPlugin, source: str = "unknown") -> PluginMetadata | None:
    """
    Validate that a plugin instance is compatible and properly implemented.

    Uses PluginMetadata for structured validation of plugin attributes.

    Args:
        plugin: The plugin instance to validate.
        source: Description of where the plugin came from (for logging).

    Returns:
        PluginMetadata if the plugin is valid and compatible, None otherwise.
    """
    # Validate required interface methods exist and are callable
    required_methods = ["can_parse", "parse_progress"]

    for method_name in required_methods:
        method = getattr(plugin, method_name, None)
        if method is None or not callable(method):
            logger.warning(
                "Plugin from %s is missing required method '%s'. Skipping.",
                source,
                method_name,
            )
            return None

    # Use PluginMetadata for structured validation
    try:
        metadata = PluginMetadata.from_plugin(plugin)
    except (ValueError, AttributeError) as e:
        logger.warning(
            "Plugin from %s failed metadata validation: %s. Skipping.",
            source,
            e,
        )
        return None

    # Check API version compatibility
    if not metadata.is_compatible(PLUGIN_API_VERSION):
        logger.warning(
            "Plugin %s from %s requires API version %d, but current version is %d. Skipping.",
            metadata.name,
            source,
            metadata.api_version,
            PLUGIN_API_VERSION,
        )
        return None

    return metadata


def load_user_plugins(
    plugin_dirs: list[Path] | None = None,
    force_reload: bool = False,
) -> list[ToolProgressPlugin]:
    """
    Load custom user plugins from plugin directories.

    User plugins are Python files in ~/.snakesee/plugins/ or ~/.config/snakesee/plugins/
    that define classes inheriting from ToolProgressPlugin.

    Args:
        plugin_dirs: List of directories to search. Defaults to USER_PLUGIN_DIRS.
        force_reload: If True, reload plugins even if already cached.

    Returns:
        List of loaded user plugin instances.

    Example plugin file (~/.snakesee/plugins/my_tool.py)::

        from snakesee.plugins.base import ToolProgress, ToolProgressPlugin
        import re

        class MyToolPlugin(ToolProgressPlugin):
            @property
            def tool_name(self) -> str:
                return "mytool"

            def can_parse(self, rule_name: str, log_content: str) -> bool:
                return "mytool" in rule_name.lower()

            def parse_progress(self, log_content: str) -> ToolProgress | None:
                match = re.search(r"Processed (\\d+) items", log_content)
                if match:
                    return ToolProgress(items_processed=int(match.group(1)), unit="items")
                return None
    """
    global _user_plugins

    if _user_plugins is not None and not force_reload:
        return _user_plugins

    if plugin_dirs is None:
        plugin_dirs = USER_PLUGIN_DIRS

    loaded_plugins: list[ToolProgressPlugin] = []

    for plugin_dir in plugin_dirs:
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            continue

        # Security checks
        _check_plugin_dir_security(plugin_dir)

        # Find all Python files in the plugin directory
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue  # Skip private modules

            try:
                plugins = _load_plugins_from_file(plugin_file)
                loaded_plugins.extend(plugins)
            except (ImportError, SyntaxError, OSError) as e:
                logger.debug("Failed to load plugin from %s: %s", plugin_file, e)
                continue

    _user_plugins = loaded_plugins
    return loaded_plugins


def _check_plugin_dir_security(plugin_dir: Path) -> None:
    """Check plugin directory for security issues and log warnings.

    Args:
        plugin_dir: Directory to check.
    """
    # Security: warn if plugin directory is a symlink
    resolved_dir = plugin_dir.resolve()
    if plugin_dir.is_symlink():
        logger.warning(
            "Plugin directory is a symlink: %s -> %s. Ensure the target is trusted.",
            plugin_dir,
            resolved_dir,
        )

    # Security: warn if plugin directory is world-writable
    # Use stat() (not lstat) to check the actual directory permissions,
    # since lstat() on symlinks returns 0o777 which would false-positive
    try:
        dir_mode = plugin_dir.stat().st_mode
        if dir_mode & stat.S_IWOTH:
            logger.warning(
                "Plugin directory is world-writable: %s. This is a security risk.",
                plugin_dir,
            )
    except OSError:
        pass


def _load_plugins_from_file(plugin_file: Path) -> list[ToolProgressPlugin]:
    """
    Load plugin classes from a Python file.

    Args:
        plugin_file: Path to the Python file.

    Returns:
        List of plugin instances found in the file.
    """
    plugins: list[ToolProgressPlugin] = []

    # Create a unique module name based on the full file path to avoid collisions
    # when different directories have files with the same name
    import hashlib

    # MD5 is used here for uniqueness only (not security), so S324 warning is safe to ignore
    path_hash = hashlib.md5(str(plugin_file.resolve()).encode()).hexdigest()[:8]  # noqa: S324
    module_name = f"snakesee_user_plugin_{plugin_file.stem}_{path_hash}"

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
    if spec is None or spec.loader is None:
        return plugins

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up on failure
        sys.modules.pop(module_name, None)
        raise

    # Find all ToolProgressPlugin subclasses in the module
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, ToolProgressPlugin)
            and obj is not ToolProgressPlugin
        ):
            try:
                plugin_instance = obj()
                # Validate the plugin before adding it
                metadata = validate_plugin(plugin_instance, str(plugin_file))
                if metadata is not None:
                    plugins.append(plugin_instance)
                    logger.debug(
                        "Loaded plugin '%s' (API v%d) from %s",
                        metadata.name,
                        metadata.api_version,
                        plugin_file,
                    )
            except (TypeError, AttributeError, RuntimeError) as e:
                logger.debug("Failed to instantiate plugin %s: %s", name, e)
                continue

    # Clean up if no valid plugins were extracted
    if not plugins:
        sys.modules.pop(module_name, None)

    return plugins


def clear_plugin_cache() -> None:
    """Clear the cached user plugins, forcing a reload on next access."""
    global _user_plugins
    _user_plugins = None
