"""Tests for the timing profile module."""

import json
import time
from pathlib import Path

import pytest

from snakesee.exceptions import ProfileNotFoundError
from snakesee.models import RuleTimingStats
from snakesee.profile import DEFAULT_PROFILE_NAME
from snakesee.profile import RuleProfile
from snakesee.profile import TimingProfile
from snakesee.profile import export_profile_from_metadata
from snakesee.profile import find_profile
from snakesee.profile import load_profile
from snakesee.profile import save_profile


class TestRuleProfile:
    """Tests for RuleProfile."""

    def test_from_stats(self) -> None:
        """Test creating RuleProfile from RuleTimingStats."""
        now = time.time()
        stats = RuleTimingStats(
            rule="align",
            durations=[100.0, 110.0, 90.0],
            timestamps=[now - 200, now - 100, now],
        )
        profile = RuleProfile.from_stats(stats)

        assert profile.rule == "align"
        assert profile.sample_count == 3
        assert profile.mean_duration == 100.0
        assert profile.durations == [100.0, 110.0, 90.0]
        assert len(profile.timestamps) == 3

    def test_to_stats(self) -> None:
        """Test converting RuleProfile back to RuleTimingStats."""
        profile = RuleProfile(
            rule="align",
            sample_count=3,
            mean_duration=100.0,
            std_dev=10.0,
            min_duration=90.0,
            max_duration=110.0,
            durations=[100.0, 110.0, 90.0],
            timestamps=[1000.0, 2000.0, 3000.0],
        )
        stats = profile.to_stats()

        assert stats.rule == "align"
        assert stats.durations == [100.0, 110.0, 90.0]
        assert stats.timestamps == [1000.0, 2000.0, 3000.0]


class TestTimingProfile:
    """Tests for TimingProfile."""

    def test_create_new(self) -> None:
        """Test creating a new profile from rule stats."""
        now = time.time()
        rule_stats = {
            "align": RuleTimingStats(
                rule="align",
                durations=[100.0, 110.0],
                timestamps=[now - 100, now],
            ),
            "sort": RuleTimingStats(
                rule="sort",
                durations=[50.0],
                timestamps=[now],
            ),
        }

        profile = TimingProfile.create_new(rule_stats)

        assert profile.version == 1
        assert "align" in profile.rules
        assert "sort" in profile.rules
        assert profile.rules["align"].sample_count == 2
        assert profile.rules["sort"].sample_count == 1
        assert profile.machine is not None  # Should be hostname

    def test_to_rule_stats(self) -> None:
        """Test converting profile back to rule stats."""
        profile = TimingProfile(
            version=1,
            created="2025-01-01T00:00:00Z",
            updated="2025-01-01T00:00:00Z",
            machine="test-machine",
            rules={
                "align": RuleProfile(
                    rule="align",
                    sample_count=2,
                    mean_duration=105.0,
                    std_dev=5.0,
                    min_duration=100.0,
                    max_duration=110.0,
                    durations=[100.0, 110.0],
                    timestamps=[1000.0, 2000.0],
                ),
            },
        )

        stats = profile.to_rule_stats()
        assert "align" in stats
        assert stats["align"].count == 2

    def test_merge_with(self) -> None:
        """Test merging two profiles."""
        now = time.time()

        profile1 = TimingProfile(
            version=1,
            created="2025-01-01T00:00:00Z",
            updated="2025-01-01T00:00:00Z",
            machine="machine1",
            rules={
                "align": RuleProfile(
                    rule="align",
                    sample_count=2,
                    mean_duration=100.0,
                    std_dev=5.0,
                    min_duration=95.0,
                    max_duration=105.0,
                    durations=[95.0, 105.0],
                    timestamps=[now - 200, now - 100],
                ),
            },
        )

        profile2 = TimingProfile(
            version=1,
            created="2025-01-02T00:00:00Z",
            updated="2025-01-02T00:00:00Z",
            machine="machine2",
            rules={
                "align": RuleProfile(
                    rule="align",
                    sample_count=1,
                    mean_duration=110.0,
                    std_dev=0.0,
                    min_duration=110.0,
                    max_duration=110.0,
                    durations=[110.0],
                    timestamps=[now],
                ),
                "sort": RuleProfile(
                    rule="sort",
                    sample_count=1,
                    mean_duration=50.0,
                    std_dev=0.0,
                    min_duration=50.0,
                    max_duration=50.0,
                    durations=[50.0],
                    timestamps=[now],
                ),
            },
        )

        merged = profile1.merge_with(profile2)

        # Should have align (3 samples) and sort (1 sample)
        assert "align" in merged.rules
        assert "sort" in merged.rules
        assert merged.rules["align"].sample_count == 3
        assert merged.rules["sort"].sample_count == 1
        # Keeps original creation time
        assert merged.created == "2025-01-01T00:00:00Z"


class TestProfileIO:
    """Tests for profile save/load functions."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading a profile."""
        now = time.time()
        profile = TimingProfile(
            version=1,
            created="2025-01-01T00:00:00Z",
            updated="2025-01-01T00:00:00Z",
            machine="test-machine",
            rules={
                "align": RuleProfile(
                    rule="align",
                    sample_count=2,
                    mean_duration=100.0,
                    std_dev=5.0,
                    min_duration=95.0,
                    max_duration=105.0,
                    durations=[95.0, 105.0],
                    timestamps=[now - 100, now],
                ),
            },
        )

        profile_path = tmp_path / "test-profile.json"
        save_profile(profile, profile_path)

        # Verify file exists and is valid JSON
        assert profile_path.exists()
        data = json.loads(profile_path.read_text())
        assert data["version"] == 1
        assert "align" in data["rules"]

        # Load it back
        loaded = load_profile(profile_path)
        assert loaded.version == 1
        assert loaded.machine == "test-machine"
        assert "align" in loaded.rules
        assert loaded.rules["align"].sample_count == 2

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading a nonexistent profile raises error."""
        with pytest.raises(ProfileNotFoundError):
            load_profile(tmp_path / "nonexistent.json")


class TestFindProfile:
    """Tests for profile discovery."""

    def test_find_in_current_dir(self, tmp_path: Path) -> None:
        """Test finding profile in current directory."""
        profile_path = tmp_path / DEFAULT_PROFILE_NAME
        profile_path.write_text("{}")

        found = find_profile(tmp_path)
        assert found == profile_path

    def test_find_in_parent_dir(self, tmp_path: Path) -> None:
        """Test finding profile in parent directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        profile_path = tmp_path / DEFAULT_PROFILE_NAME
        profile_path.write_text("{}")

        found = find_profile(subdir)
        assert found == profile_path

    def test_not_found(self, tmp_path: Path) -> None:
        """Test returns None when profile not found."""
        found = find_profile(tmp_path)
        assert found is None


class TestExportProfile:
    """Tests for profile export from metadata."""

    def test_export_from_metadata(self, metadata_dir: Path, tmp_path: Path) -> None:
        """Test exporting profile from metadata directory."""
        output_path = tmp_path / "exported.json"

        profile = export_profile_from_metadata(
            metadata_dir=metadata_dir,
            output_path=output_path,
        )

        assert output_path.exists()
        assert "align" in profile.rules
        assert "sort" in profile.rules
        assert profile.rules["align"].sample_count == 5
        assert profile.rules["sort"].sample_count == 3

    def test_export_with_merge(self, metadata_dir: Path, tmp_path: Path) -> None:
        """Test exporting with merge into existing profile."""
        output_path = tmp_path / "merged.json"

        # Create initial profile
        existing = TimingProfile(
            version=1,
            created="2025-01-01T00:00:00Z",
            updated="2025-01-01T00:00:00Z",
            machine="old-machine",
            rules={
                "old_rule": RuleProfile(
                    rule="old_rule",
                    sample_count=1,
                    mean_duration=999.0,
                    std_dev=0.0,
                    min_duration=999.0,
                    max_duration=999.0,
                    durations=[999.0],
                    timestamps=[1000.0],
                ),
            },
        )
        save_profile(existing, output_path)

        # Export with merge
        profile = export_profile_from_metadata(
            metadata_dir=metadata_dir,
            output_path=output_path,
            merge_existing=True,
        )

        # Should have both old and new rules
        assert "old_rule" in profile.rules
        assert "align" in profile.rules
        assert "sort" in profile.rules
