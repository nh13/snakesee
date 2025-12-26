"""Tests for the plugin system."""

from pathlib import Path

import pytest

from snakesee.plugins import BUILTIN_PLUGINS
from snakesee.plugins import PluginMetadata
from snakesee.plugins import find_plugin_for_log
from snakesee.plugins import find_rule_log
from snakesee.plugins import parse_tool_progress
from snakesee.plugins.base import PLUGIN_API_VERSION
from snakesee.plugins.base import ToolProgress
from snakesee.plugins.bwa import BWAPlugin
from snakesee.plugins.fastp import FastpPlugin
from snakesee.plugins.samtools import SamtoolsSortPlugin
from snakesee.plugins.star import STARPlugin


class TestToolProgress:
    """Tests for ToolProgress dataclass."""

    def test_basic_progress(self) -> None:
        """Test basic progress creation."""
        progress = ToolProgress(items_processed=1000, unit="reads")
        assert progress.items_processed == 1000
        assert progress.unit == "reads"
        assert progress.items_total is None

    def test_percent_complete_calculated(self) -> None:
        """Test percent_complete is calculated from total."""
        progress = ToolProgress(items_processed=50, items_total=100, unit="reads")
        assert progress.percent_complete == 50.0

    def test_percent_complete_explicit(self) -> None:
        """Test explicit percent_complete is preserved."""
        progress = ToolProgress(
            items_processed=50,
            items_total=100,
            unit="reads",
            percent_complete=60.0,  # Explicit override
        )
        assert progress.percent_complete == 60.0

    def test_progress_str(self) -> None:
        """Test progress string formatting."""
        progress = ToolProgress(items_processed=1000000, items_total=2000000, unit="reads")
        assert progress.progress_str == "1,000,000/2,000,000 reads"

    def test_progress_str_no_total(self) -> None:
        """Test progress string without total."""
        progress = ToolProgress(items_processed=1000000, unit="reads")
        assert progress.progress_str == "1,000,000 reads"

    def test_percent_str(self) -> None:
        """Test percent string formatting."""
        progress = ToolProgress(items_processed=50, items_total=100, unit="reads")
        assert progress.percent_str == "50.0%"

    def test_percent_str_unknown(self) -> None:
        """Test percent string when unknown."""
        progress = ToolProgress(items_processed=50, unit="reads")
        assert progress.percent_str == "?"


class TestBWAPlugin:
    """Tests for BWA plugin."""

    def test_can_parse_by_rule_name(self) -> None:
        """Test can_parse detects BWA by rule name."""
        plugin = BWAPlugin()
        assert plugin.can_parse("bwa_align", "")
        assert plugin.can_parse("bwa_mem", "")
        assert not plugin.can_parse("star_align", "")

    def test_can_parse_by_log_content(self) -> None:
        """Test can_parse detects BWA by log content."""
        plugin = BWAPlugin()
        log_content = "[M::bwa_idx_load_from_disk] read 0 ALT contigs"
        assert plugin.can_parse("align", log_content)

    def test_parse_processed_pattern(self) -> None:
        """Test parsing BWA processed reads pattern."""
        plugin = BWAPlugin()
        log_content = """
        [M::bwa_idx_load_from_disk] read 0 ALT contigs
        [M::mem_process_seqs] Processed 10000 reads in 1.234 CPU sec
        [M::mem_process_seqs] Processed 10000 reads in 1.345 CPU sec
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 20000
        assert progress.unit == "reads"

    def test_parse_read_sequences_pattern(self) -> None:
        """Test parsing BWA read sequences pattern."""
        plugin = BWAPlugin()
        log_content = """
        [M::process] read 10000 sequences (1500000 bp)...
        [M::process] read 10000 sequences (1500000 bp)...
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 20000

    def test_parse_no_progress(self) -> None:
        """Test parsing returns None when no progress found."""
        plugin = BWAPlugin()
        progress = plugin.parse_progress("Some random log content")
        assert progress is None


class TestSamtoolsSortPlugin:
    """Tests for samtools sort plugin."""

    def test_can_parse(self) -> None:
        """Test can_parse detection."""
        plugin = SamtoolsSortPlugin()
        assert plugin.can_parse("samtools_sort", "")
        assert plugin.can_parse("sort_bam", "[bam_sort_core] merging...")

    def test_parse_progress(self) -> None:
        """Test parsing samtools sort progress."""
        plugin = SamtoolsSortPlugin()
        log_content = """
        [bam_sort_core] merging from 4 files...
        read 500000 records...
        read 1000000 records...
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 1000000
        assert progress.unit == "records"


class TestFastpPlugin:
    """Tests for fastp plugin."""

    def test_can_parse(self) -> None:
        """Test can_parse detection."""
        plugin = FastpPlugin()
        assert plugin.can_parse("fastp_qc", "")
        assert plugin.can_parse("qc", "fastp version 0.23.0")

    def test_parse_total_reads(self) -> None:
        """Test parsing fastp total reads."""
        plugin = FastpPlugin()
        log_content = """
        Read1 before filtering:
        total reads: 10000000
        total bases: 1500000000
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 10000000

    def test_parse_passed_reads(self) -> None:
        """Test parsing fastp passed reads with total."""
        plugin = FastpPlugin()
        log_content = """
        Read1 before filtering:
        total reads: 10000000
        Filtering result:
        reads passed filter: 9800000
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 9800000
        assert progress.items_total == 10000000


class TestSTARPlugin:
    """Tests for STAR plugin."""

    def test_can_parse(self) -> None:
        """Test can_parse detection."""
        plugin = STARPlugin()
        assert plugin.can_parse("star_align", "")
        assert plugin.can_parse("align", "STAR version=2.7.10a")

    def test_parse_finished_reads(self) -> None:
        """Test parsing STAR finished reads."""
        plugin = STARPlugin()
        log_content = """
        STAR version=2.7.10a
        Finished 10000000 paired reads
        Finished 20000000 paired reads
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 20000000
        assert progress.unit == "reads"


class TestFgbioPlugin:
    """Tests for fgbio plugin."""

    def test_can_parse_by_rule_name(self) -> None:
        """Test can_parse detects fgbio by rule name."""
        from snakesee.plugins.fgbio import FgbioPlugin

        plugin = FgbioPlugin()
        assert plugin.can_parse("fgbio_group_reads", "")
        assert plugin.can_parse("GroupReadsByUmi", "")
        assert plugin.can_parse("call_consensus", "com.fulcrumgenomics")
        assert not plugin.can_parse("bwa_align", "")

    def test_can_parse_by_log_content(self) -> None:
        """Test can_parse detects fgbio by log content."""
        from snakesee.plugins.fgbio import FgbioPlugin

        plugin = FgbioPlugin()
        log_content = "[INFO] com.fulcrumgenomics.umi.GroupReadsByUmi"
        assert plugin.can_parse("some_rule", log_content)

    def test_parse_processed_pattern(self) -> None:
        """Test parsing fgbio processed records pattern."""
        from snakesee.plugins.fgbio import FgbioPlugin

        plugin = FgbioPlugin()
        log_content = """
        [INFO] Starting GroupReadsByUmi
        [INFO] Processed 1,000,000 records. Elapsed time: 00:01:30.
        [INFO] Processed 2,000,000 records. Elapsed time: 00:03:00.
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 2000000
        assert progress.unit == "records"

    def test_parse_read_pattern(self) -> None:
        """Test parsing fgbio read records pattern."""
        from snakesee.plugins.fgbio import FgbioPlugin

        plugin = FgbioPlugin()
        log_content = """
        [INFO] Read 5000000 records from BAM file.
        """
        progress = plugin.parse_progress(log_content)
        assert progress is not None
        assert progress.items_processed == 5000000
        assert progress.unit == "records"

    def test_parse_no_progress(self) -> None:
        """Test parsing returns None when no progress found."""
        from snakesee.plugins.fgbio import FgbioPlugin

        plugin = FgbioPlugin()
        progress = plugin.parse_progress("Some random log content")
        assert progress is None


class TestPluginRegistry:
    """Tests for plugin registry functions."""

    def test_builtin_plugins_exist(self) -> None:
        """Test that built-in plugins are registered."""
        assert len(BUILTIN_PLUGINS) >= 6
        tool_names = [p.tool_name for p in BUILTIN_PLUGINS]
        assert "bwa" in tool_names
        assert "fastp" in tool_names
        assert "fgbio" in tool_names
        assert "star" in tool_names

    def test_entry_point_discovery_returns_list(self) -> None:
        """Test that entry point discovery returns a list (even if empty)."""
        from snakesee.plugins import discover_entry_point_plugins

        plugins = discover_entry_point_plugins(force_reload=True)
        assert isinstance(plugins, list)

    def test_get_all_plugins_includes_builtin(self) -> None:
        """Test that get_all_plugins includes built-in plugins."""
        from snakesee.plugins import get_all_plugins

        all_plugins = get_all_plugins(include_user=False)
        assert len(all_plugins) >= 6
        tool_names = [p.tool_name for p in all_plugins]
        assert "bwa" in tool_names

    def test_find_plugin_for_log_bwa(self) -> None:
        """Test finding BWA plugin."""
        log_content = "[M::bwa_idx_load_from_disk] read 0 ALT contigs"
        plugin = find_plugin_for_log("align", log_content)
        assert plugin is not None
        assert plugin.tool_name == "bwa"

    def test_find_plugin_for_log_none(self) -> None:
        """Test returns None when no plugin matches."""
        plugin = find_plugin_for_log("unknown_rule", "random log content")
        assert plugin is None

    def test_parse_tool_progress(self, tmp_path: Path) -> None:
        """Test parse_tool_progress from file."""
        log_file = tmp_path / "bwa.log"
        log_file.write_text("[M::mem_process_seqs] Processed 10000 reads in 1.0 CPU sec")

        progress = parse_tool_progress("bwa_align", log_file)
        assert progress is not None
        assert progress.items_processed == 10000

    def test_parse_tool_progress_nonexistent(self, tmp_path: Path) -> None:
        """Test parse_tool_progress with nonexistent file."""
        progress = parse_tool_progress("bwa", tmp_path / "nonexistent.log")
        assert progress is None


class TestUserPluginLoading:
    """Tests for user plugin directory loading."""

    def test_load_user_plugins_empty_dir(self, tmp_path: Path) -> None:
        """Test loading from empty directory returns empty list."""
        from snakesee.plugins import load_user_plugins

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        assert plugins == []

    def test_load_user_plugins_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test loading from nonexistent directory returns empty list."""
        from snakesee.plugins import load_user_plugins

        nonexistent = tmp_path / "nonexistent"
        plugins = load_user_plugins(plugin_dirs=[nonexistent], force_reload=True)
        assert plugins == []

    def test_load_user_plugins_valid_plugin(self, tmp_path: Path) -> None:
        """Test loading a valid user plugin."""
        from snakesee.plugins import load_user_plugins

        plugin_file = tmp_path / "my_tool.py"
        plugin_file.write_text("""
import re
from snakesee.plugins.base import ToolProgress, ToolProgressPlugin

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
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        assert len(plugins) == 1
        assert plugins[0].tool_name == "mytool"

    def test_load_user_plugins_skips_private(self, tmp_path: Path) -> None:
        """Test that files starting with underscore are skipped."""
        from snakesee.plugins import load_user_plugins

        private_file = tmp_path / "_private.py"
        private_file.write_text("""
from snakesee.plugins.base import ToolProgress, ToolProgressPlugin

class PrivatePlugin(ToolProgressPlugin):
    @property
    def tool_name(self) -> str:
        return "private"

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        return False

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        return None
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        assert plugins == []

    def test_load_user_plugins_skips_invalid(self, tmp_path: Path) -> None:
        """Test that invalid plugin files are skipped."""
        from snakesee.plugins import load_user_plugins

        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("this is not valid python {{{")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        assert plugins == []

    def test_loaded_plugin_works(self, tmp_path: Path) -> None:
        """Test that a loaded user plugin actually works."""
        from snakesee.plugins import load_user_plugins

        plugin_file = tmp_path / "counter.py"
        plugin_file.write_text("""
import re
from snakesee.plugins.base import ToolProgress, ToolProgressPlugin

class CounterPlugin(ToolProgressPlugin):
    @property
    def tool_name(self) -> str:
        return "counter"

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        return "counter" in rule_name.lower() or "Count:" in log_content

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        match = re.search(r"Count: (\\d+)", log_content)
        if match:
            return ToolProgress(items_processed=int(match.group(1)), unit="items")
        return None
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        assert len(plugins) == 1

        plugin = plugins[0]
        assert plugin.can_parse("counter_step", "")
        assert plugin.can_parse("some_rule", "Count: 500")

        progress = plugin.parse_progress("Processing... Count: 12345")
        assert progress is not None
        assert progress.items_processed == 12345
        assert progress.unit == "items"


class TestFindRuleLog:
    """Tests for find_rule_log function."""

    def test_find_in_logs_dir(self, tmp_path: Path) -> None:
        """Test finding log in logs/ directory."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "bwa_align.log"
        log_file.write_text("test")

        found = find_rule_log("bwa_align", None, tmp_path)
        assert found is not None
        assert found.name == "bwa_align.log"

    def test_find_none_when_missing(self, tmp_path: Path) -> None:
        """Test returns None when no log found."""
        found = find_rule_log("unknown_rule", None, tmp_path)
        assert found is None


class TestPluginErrorHandling:
    """Tests for plugin error handling."""

    def test_load_plugins_with_runtime_error(self, tmp_path: Path) -> None:
        """Test that plugins raising errors during instantiation are skipped."""
        from snakesee.plugins import load_user_plugins

        plugin_file = tmp_path / "bad_init.py"
        plugin_file.write_text("""
from snakesee.plugins.base import ToolProgress, ToolProgressPlugin

class BadInitPlugin(ToolProgressPlugin):
    def __init__(self):
        raise RuntimeError("Init failed")

    @property
    def tool_name(self) -> str:
        return "bad"

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        return False

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        return None
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        # Plugin should be skipped due to RuntimeError
        assert plugins == []

    def test_load_plugins_with_import_error(self, tmp_path: Path) -> None:
        """Test that plugins with import errors are skipped."""
        from snakesee.plugins import load_user_plugins

        plugin_file = tmp_path / "bad_import.py"
        plugin_file.write_text("""
import nonexistent_module_xyz123
from snakesee.plugins.base import ToolProgress, ToolProgressPlugin

class BadImportPlugin(ToolProgressPlugin):
    @property
    def tool_name(self) -> str:
        return "bad"

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        return False

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        return None
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        # Plugin should be skipped due to ImportError
        assert plugins == []

    def test_parse_progress_handles_exception(self, tmp_path: Path) -> None:
        """Test that parse_tool_progress handles plugin exceptions gracefully."""
        from snakesee.plugins import parse_tool_progress

        # Create a log file
        log_file = tmp_path / "test.log"
        log_file.write_text("Some log content")

        # This shouldn't raise, even if no plugin matches
        result = parse_tool_progress("unknown_rule", log_file)
        assert result is None

    def test_find_rule_log_handles_missing_dirs(self, tmp_path: Path) -> None:
        """Test that find_rule_log handles missing directories gracefully."""
        # Non-existent workflow dir
        result = find_rule_log("test_rule", None, tmp_path / "nonexistent")
        assert result is None

    def test_load_plugins_skips_module_without_plugin_class(self, tmp_path: Path) -> None:
        """Test that modules without ToolProgressPlugin subclasses are skipped."""
        from snakesee.plugins import load_user_plugins

        # Create a valid Python file but without a plugin class
        plugin_file = tmp_path / "not_a_plugin.py"
        plugin_file.write_text("""
def some_function():
    pass

class NotAPlugin:
    pass
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        # No plugin should be loaded
        assert plugins == []

    def test_load_plugins_with_type_error(self, tmp_path: Path) -> None:
        """Test that plugins with TypeError during instantiation are skipped."""
        from snakesee.plugins import load_user_plugins

        plugin_file = tmp_path / "type_error.py"
        plugin_file.write_text("""
from snakesee.plugins.base import ToolProgress, ToolProgressPlugin

class TypeErrorPlugin(ToolProgressPlugin):
    def __init__(self, required_arg):  # Requires an arg we don't provide
        pass

    @property
    def tool_name(self) -> str:
        return "type_error"

    def can_parse(self, rule_name: str, log_content: str) -> bool:
        return False

    def parse_progress(self, log_content: str) -> ToolProgress | None:
        return None
""")

        plugins = load_user_plugins(plugin_dirs=[tmp_path], force_reload=True)
        # Plugin should be skipped due to TypeError (missing required arg)
        assert plugins == []


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass and validation."""

    def test_create_basic_metadata(self) -> None:
        """Test creating basic metadata."""
        metadata = PluginMetadata(name="test", api_version=1)
        assert metadata.name == "test"
        assert metadata.api_version == 1
        assert metadata.patterns == ()
        assert metadata.description == ""

    def test_create_full_metadata(self) -> None:
        """Test creating metadata with all fields."""
        metadata = PluginMetadata(
            name="bwa",
            api_version=1,
            patterns=("bwa_align", "bwa_mem"),
            description="BWA alignment plugin",
        )
        assert metadata.name == "bwa"
        assert metadata.patterns == ("bwa_align", "bwa_mem")
        assert metadata.description == "BWA alignment plugin"

    def test_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Plugin name cannot be empty"):
            PluginMetadata(name="", api_version=1)

    def test_whitespace_name_raises(self) -> None:
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Plugin name cannot be empty"):
            PluginMetadata(name="   ", api_version=1)

    def test_invalid_api_version_raises(self) -> None:
        """Test that API version < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Plugin API version must be >= 1"):
            PluginMetadata(name="test", api_version=0)

        with pytest.raises(ValueError, match="Plugin API version must be >= 1"):
            PluginMetadata(name="test", api_version=-1)

    def test_from_plugin_bwa(self) -> None:
        """Test creating metadata from BWA plugin."""
        plugin = BWAPlugin()
        metadata = PluginMetadata.from_plugin(plugin)
        assert metadata.name == "bwa"
        assert metadata.api_version >= 1
        assert len(metadata.patterns) > 0

    def test_from_plugin_fastp(self) -> None:
        """Test creating metadata from fastp plugin."""
        plugin = FastpPlugin()
        metadata = PluginMetadata.from_plugin(plugin)
        assert metadata.name == "fastp"

    def test_from_plugin_star(self) -> None:
        """Test creating metadata from STAR plugin."""
        plugin = STARPlugin()
        metadata = PluginMetadata.from_plugin(plugin)
        assert metadata.name == "star"

    def test_is_compatible_same_version(self) -> None:
        """Test is_compatible with same API version."""
        metadata = PluginMetadata(name="test", api_version=1)
        assert metadata.is_compatible(1) is True

    def test_is_compatible_older_version(self) -> None:
        """Test is_compatible with older plugin version."""
        metadata = PluginMetadata(name="test", api_version=1)
        # Plugin v1 is compatible with API v2
        assert metadata.is_compatible(2) is True

    def test_is_compatible_newer_version(self) -> None:
        """Test is_compatible with newer plugin version."""
        metadata = PluginMetadata(name="test", api_version=2)
        # Plugin v2 is NOT compatible with API v1
        assert metadata.is_compatible(1) is False

    def test_is_compatible_default_version(self) -> None:
        """Test is_compatible with default (current) API version."""
        metadata = PluginMetadata(name="test", api_version=PLUGIN_API_VERSION)
        assert metadata.is_compatible() is True

    def test_metadata_is_frozen(self) -> None:
        """Test that PluginMetadata is immutable."""
        metadata = PluginMetadata(name="test", api_version=1)
        with pytest.raises(AttributeError):
            metadata.name = "modified"  # type: ignore[misc]


class TestPluginValidation:
    """Tests for plugin validation with PluginMetadata."""

    def test_validate_plugin_returns_metadata(self) -> None:
        """Test that validate_plugin returns PluginMetadata on success."""
        from snakesee.plugins.loader import validate_plugin

        plugin = BWAPlugin()
        result = validate_plugin(plugin, "test")
        assert result is not None
        assert isinstance(result, PluginMetadata)
        assert result.name == "bwa"

    def test_validate_plugin_missing_method(self) -> None:
        """Test that plugin missing required methods returns None."""
        from snakesee.plugins.loader import validate_plugin

        # Create a class that's not a proper plugin
        class BadPlugin:
            @property
            def tool_name(self) -> str:
                return "bad"

            # Missing can_parse and parse_progress

        result = validate_plugin(BadPlugin(), "test")  # type: ignore[arg-type]
        assert result is None

    def test_validate_all_builtin_plugins(self) -> None:
        """Test that all builtin plugins pass validation."""
        from snakesee.plugins.loader import validate_plugin

        for plugin in BUILTIN_PLUGINS:
            result = validate_plugin(plugin, "builtin")
            assert result is not None, f"Plugin {plugin.tool_name} failed validation"
            assert result.name == plugin.tool_name
