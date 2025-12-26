"""Plugin system for tool-specific progress parsing.

This package provides a plugin system for parsing progress from tool-specific
log files. Plugins can detect and parse output from common bioinformatics tools
like BWA, samtools, STAR, etc.

Plugin Discovery:
    Plugins are discovered from three sources:
    1. Built-in plugins shipped with snakesee
    2. User plugins in ~/.snakesee/plugins/ or ~/.config/snakesee/plugins/
    3. Entry points registered by third-party packages

See Also:
    - :mod:`snakesee.plugins.loader` for file-based plugin loading
    - :mod:`snakesee.plugins.discovery` for entry point discovery
    - :mod:`snakesee.plugins.registry` for plugin lookup functions
"""

import stat
from pathlib import Path

import orjson

from snakesee.plugins.base import PluginMetadata
from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin
from snakesee.plugins.bwa import BWAPlugin
from snakesee.plugins.discovery import ENTRY_POINT_GROUP
from snakesee.plugins.discovery import discover_entry_point_plugins
from snakesee.plugins.fastp import FastpPlugin
from snakesee.plugins.fgbio import FgbioPlugin
from snakesee.plugins.loader import USER_PLUGIN_DIRS
from snakesee.plugins.loader import load_user_plugins
from snakesee.plugins.registry import find_plugin_for_log as _find_plugin_for_log
from snakesee.plugins.registry import get_all_plugins as _get_all_plugins
from snakesee.plugins.registry import parse_tool_progress as _parse_tool_progress
from snakesee.plugins.samtools import SamtoolsIndexPlugin
from snakesee.plugins.samtools import SamtoolsSortPlugin
from snakesee.plugins.star import STARPlugin

__all__ = [
    "PluginMetadata",
    "ToolProgress",
    "ToolProgressPlugin",
    "BUILTIN_PLUGINS",
    "ENTRY_POINT_GROUP",
    "USER_PLUGIN_DIRS",
    "find_plugin_for_log",
    "parse_tool_progress",
    "load_user_plugins",
    "discover_entry_point_plugins",
    "get_all_plugins",
    "find_rule_log",
]

# Built-in plugins for common bioinformatics tools
BUILTIN_PLUGINS: list[ToolProgressPlugin] = [
    BWAPlugin(),
    SamtoolsSortPlugin(),
    SamtoolsIndexPlugin(),
    FastpPlugin(),
    FgbioPlugin(),
    STARPlugin(),
]


def get_all_plugins(include_user: bool = True) -> list[ToolProgressPlugin]:
    """
    Get all available plugins (built-in, user file-based, and entry points).

    Args:
        include_user: Whether to include user plugins (file-based and entry points).

    Returns:
        Combined list of all plugins.
    """
    return _get_all_plugins(BUILTIN_PLUGINS, include_user)


def find_plugin_for_log(
    rule_name: str,
    log_content: str,
    plugins: list[ToolProgressPlugin] | None = None,
) -> ToolProgressPlugin | None:
    """
    Find a plugin that can parse the given log content.

    Args:
        rule_name: Name of the Snakemake rule.
        log_content: Content of the rule's log file.
        plugins: List of plugins to search. Defaults to all plugins.

    Returns:
        A plugin that can parse this log, or None if no plugin matches.
    """
    if plugins is None:
        plugins = get_all_plugins()
    return _find_plugin_for_log(rule_name, log_content, plugins)


def parse_tool_progress(
    rule_name: str,
    log_path: Path,
    plugins: list[ToolProgressPlugin] | None = None,
) -> ToolProgress | None:
    """
    Parse progress from a rule's log file using available plugins.

    Args:
        rule_name: Name of the Snakemake rule.
        log_path: Path to the rule's log file.
        plugins: List of plugins to use. Defaults to all plugins (built-in + user).

    Returns:
        ToolProgress if progress could be extracted, None otherwise.
    """
    if plugins is None:
        plugins = get_all_plugins()
    return _parse_tool_progress(rule_name, log_path, plugins)


def _search_log_dir(
    log_dir: Path,
    rule_name: str,
    wildcards: dict[str, str] | None,
) -> list[Path]:
    """Search a log directory for logs matching rule_name and wildcards."""
    paths: list[Path] = []
    if not log_dir.exists():
        return paths
    paths.extend(log_dir.glob(f"**/{rule_name}*"))
    rule_log_dir = log_dir / rule_name
    if rule_log_dir.exists():
        paths.extend(rule_log_dir.glob("*"))
    if wildcards:
        for wc_value in wildcards.values():
            if wc_value:
                paths.extend(log_dir.glob(f"**/*{wc_value}*"))
    return paths


def find_rule_log(
    rule_name: str,
    job_id: int | str | None,
    workflow_dir: Path,
    wildcards: dict[str, str] | None = None,
) -> Path | None:
    """
    Attempt to find the log file for a running rule.

    Snakemake stores rule logs in various locations depending on the
    workflow configuration. This function searches common locations.

    Args:
        rule_name: Name of the rule.
        job_id: Snakemake job ID (if known).
        workflow_dir: Workflow root directory.
        wildcards: Dictionary of wildcard names to values for the job.

    Returns:
        Path to the log file if found, None otherwise.
    """
    snakemake_dir = workflow_dir / ".snakemake"

    # Common log locations to search
    search_paths: list[Path] = []

    # First, try to find log path from .snakemake/metadata (most reliable)
    metadata_dir = snakemake_dir / "metadata"
    if metadata_dir.exists():
        for meta_file in metadata_dir.iterdir():
            try:
                data = orjson.loads(meta_file.read_bytes())
                if data.get("rule") == rule_name and data.get("log"):
                    # Get the most recent log file for this rule
                    for log_entry in data["log"]:
                        log_path = workflow_dir / log_entry
                        if log_path.exists():
                            search_paths.append(log_path)
            except (orjson.JSONDecodeError, OSError, KeyError):
                continue

    # .snakemake/log/ directory for rule-specific logs
    log_dir = snakemake_dir / "log"
    if log_dir.exists():
        # Look for logs matching the rule name
        search_paths.extend(log_dir.glob(f"*{rule_name}*"))
        search_paths.extend(log_dir.glob(f"*job{job_id}*"))

    # logs/ directory (common convention)
    logs_dir = workflow_dir / "logs"
    search_paths.extend(_search_log_dir(logs_dir, rule_name, wildcards))

    # log/ directory (another common convention)
    search_paths.extend(_search_log_dir(workflow_dir / "log", rule_name, wildcards))

    # Deduplicate and filter to existing files, then sort by mtime
    seen: set[Path] = set()
    valid_logs: list[tuple[Path, float]] = []
    for p in search_paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            stat_result = p.stat()
            if stat.S_ISREG(stat_result.st_mode):
                valid_logs.append((p, stat_result.st_mtime))
        except OSError:
            continue

    if valid_logs:
        # Sort by mtime (newest first) and return
        valid_logs.sort(key=lambda x: x[1], reverse=True)
        return valid_logs[0][0]

    return None
