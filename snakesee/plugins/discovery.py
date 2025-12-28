"""Entry point-based plugin discovery.

This module handles discovering plugins registered via setuptools entry points.
Third-party packages can register plugins in their pyproject.toml.
"""

import logging
from importlib.metadata import distributions
from importlib.metadata import entry_points

from snakesee.plugins.base import ToolProgressPlugin
from snakesee.plugins.loader import validate_plugin

logger = logging.getLogger(__name__)

# Entry point group for third-party plugins
ENTRY_POINT_GROUP = "snakesee.plugins"

# Cache for entry point plugins
_entry_point_plugins: list[ToolProgressPlugin] | None = None
# Version hash to detect package updates
_entry_point_version_hash: int = 0


def _compute_version_hash() -> int:
    """Compute a hash of installed package versions to detect updates.

    Returns:
        Hash of (name, version) tuples, or 0 on error.
    """
    try:
        return hash(
            tuple(
                (d.metadata["Name"], d.metadata["Version"])
                for d in distributions()
                if "Name" in d.metadata and "Version" in d.metadata
            )
        )
    except (TypeError, OSError) as e:
        logger.debug("Failed to compute package version hash: %s", e)
        return 0


def discover_entry_point_plugins(
    force_reload: bool = False,
) -> list[ToolProgressPlugin]:
    """
    Discover plugins registered via setuptools entry points.

    Third-party packages can register plugins by adding an entry point
    in their pyproject.toml:

        [project.entry-points."snakesee.plugins"]
        my_tool = "my_package.plugins:MyToolPlugin"

    Args:
        force_reload: If True, re-discover plugins even if cached.

    Returns:
        List of discovered plugin instances.
    """
    global _entry_point_plugins
    global _entry_point_version_hash

    version_hash = _compute_version_hash()

    # Return cached plugins if valid and not forcing reload
    if (
        _entry_point_plugins is not None
        and not force_reload
        and _entry_point_version_hash == version_hash
    ):
        return _entry_point_plugins

    plugins: list[ToolProgressPlugin] = []

    try:
        # Python 3.10+ style
        eps = entry_points(group=ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                plugin_class = ep.load()
                if isinstance(plugin_class, type) and issubclass(plugin_class, ToolProgressPlugin):
                    plugin_instance = plugin_class()
                    # Validate the plugin before adding it
                    metadata = validate_plugin(plugin_instance, f"entry_point:{ep.name}")
                    if metadata is not None:
                        plugins.append(plugin_instance)
            except (ImportError, TypeError, AttributeError) as e:
                logger.debug("Failed to load entry point plugin %s: %s", ep.name, e)
                continue
    except (TypeError, OSError) as e:
        logger.debug("Error discovering entry points: %s", e)

    _entry_point_plugins = plugins
    _entry_point_version_hash = version_hash
    return plugins


def clear_discovery_cache() -> None:
    """Clear the cached entry point plugins, forcing a rediscovery on next access."""
    global _entry_point_plugins
    global _entry_point_version_hash
    _entry_point_plugins = None
    _entry_point_version_hash = 0
