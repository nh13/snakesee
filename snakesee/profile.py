"""Portable timing profile storage for cross-run/machine estimation."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from snakesee.models import RuleTimingStats

PROFILE_VERSION = 1
DEFAULT_PROFILE_NAME = ".snakesee-profile.json"


@dataclass
class RuleProfile:
    """
    Timing profile for a single rule.

    Attributes:
        rule: The rule name.
        sample_count: Number of executions observed.
        mean_duration: Mean duration in seconds.
        std_dev: Standard deviation of durations.
        min_duration: Minimum observed duration.
        max_duration: Maximum observed duration.
        durations: Raw duration values for merging.
        timestamps: Unix timestamps when each run completed.
    """

    rule: str
    sample_count: int
    mean_duration: float
    std_dev: float
    min_duration: float
    max_duration: float
    durations: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    @classmethod
    def from_stats(cls, stats: RuleTimingStats) -> RuleProfile:
        """Create a RuleProfile from RuleTimingStats."""
        return cls(
            rule=stats.rule,
            sample_count=stats.count,
            mean_duration=stats.mean_duration,
            std_dev=stats.std_dev,
            min_duration=stats.min_duration,
            max_duration=stats.max_duration,
            durations=list(stats.durations),
            timestamps=list(stats.timestamps),
        )

    def to_stats(self) -> RuleTimingStats:
        """Convert back to RuleTimingStats."""
        return RuleTimingStats(
            rule=self.rule,
            durations=list(self.durations),
            timestamps=list(self.timestamps),
        )


@dataclass
class TimingProfile:
    """
    Complete timing profile for a workflow.

    Attributes:
        version: Profile format version.
        created: ISO timestamp when profile was first created.
        updated: ISO timestamp when profile was last updated.
        machine: Optional machine identifier.
        rules: Dictionary of rule profiles.
    """

    version: int
    created: str
    updated: str
    rules: dict[str, RuleProfile]
    machine: str | None = None

    @classmethod
    def create_new(cls, rule_stats: dict[str, RuleTimingStats]) -> TimingProfile:
        """Create a new profile from rule timing stats."""
        now = datetime.now(timezone.utc).isoformat()
        rules = {name: RuleProfile.from_stats(stats) for name, stats in rule_stats.items()}
        return cls(
            version=PROFILE_VERSION,
            created=now,
            updated=now,
            machine=socket.gethostname(),
            rules=rules,
        )

    def to_rule_stats(self) -> dict[str, RuleTimingStats]:
        """Convert profile back to rule timing stats."""
        return {name: profile.to_stats() for name, profile in self.rules.items()}

    def merge_with(self, other: TimingProfile) -> TimingProfile:
        """
        Merge another profile into this one.

        Combines durations and timestamps from both profiles, keeping all data
        for weighted estimation. Updates the 'updated' timestamp.

        Args:
            other: Another profile to merge.

        Returns:
            A new merged profile.
        """
        merged_rules: dict[str, RuleProfile] = {}

        # Start with rules from self
        all_rule_names = set(self.rules.keys()) | set(other.rules.keys())

        for rule_name in all_rule_names:
            self_profile = self.rules.get(rule_name)
            other_profile = other.rules.get(rule_name)

            if self_profile is None and other_profile is not None:
                merged_rules[rule_name] = other_profile
            elif other_profile is None and self_profile is not None:
                merged_rules[rule_name] = self_profile
            elif self_profile is not None and other_profile is not None:
                # Merge durations and timestamps, then sort by timestamp
                combined = list(zip(self_profile.durations, self_profile.timestamps, strict=True))
                combined.extend(zip(other_profile.durations, other_profile.timestamps, strict=True))
                combined.sort(key=lambda x: x[1])  # Sort by timestamp

                durations = [x[0] for x in combined]
                timestamps = [x[1] for x in combined]

                # Create merged stats to get computed properties
                merged_stats = RuleTimingStats(
                    rule=rule_name,
                    durations=durations,
                    timestamps=timestamps,
                )

                merged_rules[rule_name] = RuleProfile(
                    rule=rule_name,
                    sample_count=merged_stats.count,
                    mean_duration=merged_stats.mean_duration,
                    std_dev=merged_stats.std_dev,
                    min_duration=merged_stats.min_duration,
                    max_duration=merged_stats.max_duration,
                    durations=durations,
                    timestamps=timestamps,
                )

        now = datetime.now(timezone.utc).isoformat()
        return TimingProfile(
            version=PROFILE_VERSION,
            created=self.created,  # Keep original creation time
            updated=now,
            machine=self.machine,  # Keep original machine
            rules=merged_rules,
        )


def save_profile(profile: TimingProfile, path: Path) -> None:
    """
    Save a timing profile to a JSON file.

    Args:
        profile: The profile to save.
        path: Path to write the profile to.
    """

    def serialize(obj: object) -> dict[str, Any]:
        if isinstance(obj, RuleProfile):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    data = {
        "version": profile.version,
        "created": profile.created,
        "updated": profile.updated,
        "machine": profile.machine,
        "rules": {name: asdict(rp) for name, rp in profile.rules.items()},
    }

    # Write atomically: write to temp file, then rename
    # This prevents corruption if the program crashes mid-write
    content = json.dumps(data, indent=2, default=serialize)
    fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.name,
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, path)
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def load_profile(path: Path) -> TimingProfile:
    """
    Load a timing profile from a JSON file.

    Args:
        path: Path to the profile file.

    Returns:
        The loaded profile.

    Raises:
        FileNotFoundError: If the profile doesn't exist.
        ValueError: If the profile format is invalid.
    """
    data = json.loads(path.read_text())

    version = data.get("version", 1)
    if version > PROFILE_VERSION:
        raise ValueError(
            f"Profile version {version} is newer than supported version {PROFILE_VERSION}"
        )

    rules = {}
    for name, rule_data in data.get("rules", {}).items():
        rules[name] = RuleProfile(
            rule=rule_data["rule"],
            sample_count=rule_data["sample_count"],
            mean_duration=rule_data["mean_duration"],
            std_dev=rule_data["std_dev"],
            min_duration=rule_data["min_duration"],
            max_duration=rule_data["max_duration"],
            durations=rule_data.get("durations", []),
            timestamps=rule_data.get("timestamps", []),
        )

    return TimingProfile(
        version=version,
        created=data["created"],
        updated=data["updated"],
        machine=data.get("machine"),
        rules=rules,
    )


def find_profile(workflow_dir: Path) -> Path | None:
    """
    Search for a profile file in the workflow directory and parent directories.

    Searches for .snakesee-profile.json in:
    1. workflow_dir
    2. Parent directories (up to 5 levels)

    Args:
        workflow_dir: Starting directory to search from.

    Returns:
        Path to the found profile, or None if not found.
    """
    current = workflow_dir.resolve()
    for _ in range(6):  # Current + 5 parent levels
        profile_path = current / DEFAULT_PROFILE_NAME
        if profile_path.exists():
            return profile_path
        if current.parent == current:
            break
        current = current.parent
    return None


def export_profile_from_metadata(
    metadata_dir: Path,
    output_path: Path | None = None,
    merge_existing: bool = False,
) -> TimingProfile:
    """
    Export a timing profile from Snakemake metadata.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        output_path: Path to write the profile. If None, returns without saving.
        merge_existing: If True and output_path exists, merge with existing profile.

    Returns:
        The created/merged profile.
    """
    from snakesee.parser import collect_rule_timing_stats

    rule_stats = collect_rule_timing_stats(metadata_dir)
    new_profile = TimingProfile.create_new(rule_stats)

    if merge_existing and output_path is not None and output_path.exists():
        existing = load_profile(output_path)
        new_profile = existing.merge_with(new_profile)

    if output_path is not None:
        save_profile(new_profile, output_path)

    return new_profile
