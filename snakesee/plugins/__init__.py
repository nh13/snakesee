"""Plugin system for tool-specific progress parsing."""

import importlib.util
import logging
import stat
import sys
from importlib.metadata import entry_points
from pathlib import Path

from snakesee.plugins.base import ToolProgress
from snakesee.plugins.base import ToolProgressPlugin
from snakesee.plugins.bwa import BWAPlugin
from snakesee.plugins.fastp import FastpPlugin
from snakesee.plugins.fgbio import FgbioPlugin
from snakesee.plugins.samtools import SamtoolsIndexPlugin
from snakesee.plugins.samtools import SamtoolsSortPlugin
from snakesee.plugins.star import STARPlugin

logger = logging.getLogger(__name__)

__all__ = [
    "ToolProgress",
    "ToolProgressPlugin",
    "BUILTIN_PLUGINS",
    "ENTRY_POINT_GROUP",
    "find_plugin_for_log",
    "parse_tool_progress",
    "load_user_plugins",
    "discover_entry_point_plugins",
    "get_all_plugins",
]

# Entry point group for third-party plugins
ENTRY_POINT_GROUP = "snakesee.plugins"

# Built-in plugins for common bioinformatics tools
BUILTIN_PLUGINS: list[ToolProgressPlugin] = [
    BWAPlugin(),
    SamtoolsSortPlugin(),
    SamtoolsIndexPlugin(),
    FastpPlugin(),
    FgbioPlugin(),
    STARPlugin(),
]

# User plugin directories (searched in order)
USER_PLUGIN_DIRS: list[Path] = [
    Path.home() / ".snakesee" / "plugins",
    Path.home() / ".config" / "snakesee" / "plugins",
]

# Cache for loaded user plugins
_user_plugins: list[ToolProgressPlugin] | None = None


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


def _load_plugins_from_file(plugin_file: Path) -> list[ToolProgressPlugin]:
    """
    Load plugin classes from a Python file.

    Args:
        plugin_file: Path to the Python file.

    Returns:
        List of plugin instances found in the file.
    """
    plugins: list[ToolProgressPlugin] = []

    # Create a unique module name based on the file path
    module_name = f"snakesee_user_plugin_{plugin_file.stem}"

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
                plugins.append(obj())
            except (TypeError, AttributeError, RuntimeError) as e:
                logger.debug("Failed to instantiate plugin %s: %s", name, e)
                continue

    return plugins


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

    if _entry_point_plugins is not None and not force_reload:
        return _entry_point_plugins

    plugins: list[ToolProgressPlugin] = []

    try:
        # Python 3.10+ style
        eps = entry_points(group=ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                plugin_class = ep.load()
                if isinstance(plugin_class, type) and issubclass(plugin_class, ToolProgressPlugin):
                    plugins.append(plugin_class())
            except (ImportError, TypeError, AttributeError) as e:
                logger.debug("Failed to load entry point plugin %s: %s", ep.name, e)
                continue
    except (TypeError, OSError) as e:
        logger.debug("Error discovering entry points: %s", e)

    _entry_point_plugins = plugins
    return plugins


# Cache for entry point plugins
_entry_point_plugins: list[ToolProgressPlugin] | None = None


def get_all_plugins(include_user: bool = True) -> list[ToolProgressPlugin]:
    """
    Get all available plugins (built-in, user file-based, and entry points).

    Args:
        include_user: Whether to include user plugins (file-based and entry points).

    Returns:
        Combined list of all plugins.
    """
    all_plugins = list(BUILTIN_PLUGINS)
    if include_user:
        all_plugins.extend(load_user_plugins())
        all_plugins.extend(discover_entry_point_plugins())
    return all_plugins


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
        plugins: List of plugins to search. Defaults to all plugins (built-in + user).

    Returns:
        A plugin that can parse this log, or None if no plugin matches.
    """
    if plugins is None:
        plugins = get_all_plugins()

    for plugin in plugins:
        if plugin.can_parse(rule_name, log_content):
            return plugin

    return None


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
    import json

    snakemake_dir = workflow_dir / ".snakemake"

    # Common log locations to search
    search_paths: list[Path] = []

    # First, try to find log path from .snakemake/metadata (most reliable)
    metadata_dir = snakemake_dir / "metadata"
    if metadata_dir.exists():
        for meta_file in metadata_dir.iterdir():
            try:
                data = json.loads(meta_file.read_text())
                if data.get("rule") == rule_name and data.get("log"):
                    # Get the most recent log file for this rule
                    for log_entry in data["log"]:
                        log_path = workflow_dir / log_entry
                        if log_path.exists():
                            search_paths.append(log_path)
            except (json.JSONDecodeError, OSError, KeyError):
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
