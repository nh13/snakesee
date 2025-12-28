"""Plugin registry and lookup functions.

This module provides functions for finding and using plugins to parse
tool-specific progress from log files.
"""

from pathlib import Path

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin


def get_all_plugins(
    builtin_plugins: list[ToolProgressPlugin],
    include_user: bool = True,
) -> list[ToolProgressPlugin]:
    """
    Get all available plugins (built-in, user file-based, and entry points).

    Args:
        builtin_plugins: List of built-in plugin instances.
        include_user: Whether to include user plugins (file-based and entry points).

    Returns:
        Combined list of all plugins.
    """
    from snakesee.plugins.discovery import discover_entry_point_plugins
    from snakesee.plugins.loader import load_user_plugins

    all_plugins = list(builtin_plugins)
    if include_user:
        all_plugins.extend(load_user_plugins())
        all_plugins.extend(discover_entry_point_plugins())
    return all_plugins


def find_plugin_for_log(
    rule_name: str,
    log_content: str,
    plugins: list[ToolProgressPlugin],
) -> ToolProgressPlugin | None:
    """
    Find a plugin that can parse the given log content.

    Args:
        rule_name: Name of the Snakemake rule.
        log_content: Content of the rule's log file.
        plugins: List of plugins to search.

    Returns:
        A plugin that can parse this log, or None if no plugin matches.
    """
    for plugin in plugins:
        if plugin.can_parse(rule_name, log_content):
            return plugin

    return None


def parse_tool_progress(
    rule_name: str,
    log_path: Path,
    plugins: list[ToolProgressPlugin],
) -> ToolProgress | None:
    """
    Parse progress from a rule's log file using available plugins.

    Args:
        rule_name: Name of the Snakemake rule.
        log_path: Path to the rule's log file.
        plugins: List of plugins to use.

    Returns:
        ToolProgress if progress could be extracted, None otherwise.
    """
    if not log_path.exists():
        return None

    try:
        content = log_path.read_text(errors="ignore")
    except OSError:
        return None

    plugin = find_plugin_for_log(rule_name, content, plugins)
    if plugin is None:
        return None

    return plugin.parse_progress(content)
