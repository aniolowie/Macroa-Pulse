# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Fingerprint , see docs/ARCHITECTURE.md §Cold Start: Module Fingerprint

A ModuleFingerprint is the structured description a module provides at
registration time. It serves two purposes:

1. Runtime routing: tells Retina which directories and memory namespaces to
   watch, and tells Layer 3 which question template to use.

2. Cold-start prior: tells Layer 2 (limbic) which feature vector slots matter
   for this module and what values to expect, so the cluster model is not
   random on day one.

Feature vector slot layout (FEATURE_DIM = 16, defined in pulse.retina):
    [0]  magnitude
    [1]  delta_type encoded
    [2]  source encoded
    [3]  hour_sin
    [4]  hour_cos
    [5]  dow_sin
    [6]  dow_cos
    [7]  minutes_since_last_activation
    [8]  size_bytes log-normalised
    [9]  directory_depth normalised
    [10] file extension hash
    [11–15] reserved (memory / network, future)
"""

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pulse.retina import FEATURE_DIM


# ---------------------------------------------------------------------------
# Sub-dataclasses for the three signal-prior categories
# ---------------------------------------------------------------------------

@dataclass
class FilesystemPrior:
    """Filesystem-specific priors declared by the module."""
    watch_directories: list[str]    # expanded (~ resolved) absolute paths
    relevant_extensions: list[str]  # e.g. [".pdf", ".docx"]
    irrelevant_extensions: list[str]


@dataclass
class MemoryPrior:
    """Memory-namespace-specific priors declared by the module."""
    watch_namespaces: list[str]      # e.g. ["/mem/homework/"]
    high_relevance_keys: list[str]   # e.g. ["due_date"]


@dataclass
class TimePrior:
    """Time-based priors declared by the module."""
    active_hours: tuple[int, int]   # (start_hour, end_hour), both 0–23 inclusive
    active_days: list[int]          # 0=Monday … 6=Sunday
    typical_interval_hours: float   # expected gap between activations


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModuleFingerprint:
    """
    Parsed and validated representation of a module's signal fingerprint.

    Constructed by parse_fingerprint(); never build directly from raw dicts.
    """
    module_id: str
    cluster: str
    version: str
    question_template: str
    default_threshold: float        # 0.0–1.0

    filesystem: Optional[FilesystemPrior] = field(default=None)
    memory: Optional[MemoryPrior] = field(default=None)
    time: Optional[TimePrior] = field(default=None)

    # ------------------------------------------------------------------
    # Convenience accessors used by Retina and Layer 3
    # ------------------------------------------------------------------

    def watch_directories(self) -> list[str]:
        """All expanded watch directories declared by this module."""
        if self.filesystem is None:
            return []
        return list(self.filesystem.watch_directories)

    def watch_namespaces(self) -> list[str]:
        """All memory namespaces declared by this module."""
        if self.memory is None:
            return []
        return list(self.memory.watch_namespaces)

    def relevant_extension_hashes(self) -> list[float]:
        """
        CRC32-based hash values for each relevant extension, normalised to
        [0.0, 1.0) using the same formula as SignalEvent.to_feature_vector
        slot [10]. Used by limbic to build positive synthetic examples.
        """
        if self.filesystem is None:
            return []
        return [
            (zlib.crc32(ext.encode()) % 1000) / 1000.0
            for ext in self.filesystem.relevant_extensions
        ]

    def irrelevant_extension_hashes(self) -> list[float]:
        """Same encoding for irrelevant extensions. Used to build negative examples."""
        if self.filesystem is None:
            return []
        return [
            (zlib.crc32(ext.encode()) % 1000) / 1000.0
            for ext in self.filesystem.irrelevant_extensions
        ]

    # ------------------------------------------------------------------
    # Feature-space description for Layer 2 cold-start
    # ------------------------------------------------------------------

    def slot_relevance_mask(self) -> np.ndarray:
        """
        Returns a float32 array of length FEATURE_DIM where each value is
        in [0.0, 1.0] and indicates how much this module cares about that
        feature slot. Used by limbic to initialise model weights with a
        meaningful prior instead of random noise.

        A value of 1.0 means the slot is directly relevant.
        A value of 0.5 means the slot is weakly relevant or context-dependent.
        A value of 0.0 means the slot is not relevant to this module.
        """
        mask = np.zeros(FEATURE_DIM, dtype=np.float32)

        # [0] magnitude: relevant for all modules
        mask[0] = 1.0

        # [1] delta_type: highly relevant when filesystem events are expected
        mask[1] = 1.0 if self.filesystem is not None else 0.5

        # [2] source: always relevant (distinguishes event types)
        mask[2] = 1.0

        # [3–6] temporal cyclical features: relevant when time priors exist
        if self.time is not None:
            has_hour_pref = self.time.active_hours != (0, 23)
            has_day_pref = len(self.time.active_days) < 7
            hour_weight = 1.0 if has_hour_pref else 0.5
            day_weight = 1.0 if has_day_pref else 0.5
            mask[3] = hour_weight   # hour_sin
            mask[4] = hour_weight   # hour_cos
            mask[5] = day_weight    # dow_sin
            mask[6] = day_weight    # dow_cos

        # [7] minutes_since_last_activation: relevant when typical interval is declared
        if self.time is not None:
            mask[7] = 1.0

        # [8–10] filesystem features
        if self.filesystem is not None:
            mask[8] = 1.0   # size_bytes log-normalised
            mask[9] = 1.0   # directory_depth normalised
            mask[10] = 1.0 if self.filesystem.relevant_extensions else 0.5

        # [11–15] reserved (memory/network, not yet implemented)
        # mask[11:16] remains 0.0

        return mask

    def active_hour_range_encoded(self) -> Optional[tuple[float, float, float, float]]:
        """
        Returns (start_sin, start_cos, end_sin, end_cos) for the declared
        active hour window, or None if no time prior is set. Used by limbic
        to seed time-based synthetic examples.
        """
        if self.time is None:
            return None
        start, end = self.time.active_hours
        return (
            math.sin(2 * math.pi * start / 24),
            math.cos(2 * math.pi * start / 24),
            math.sin(2 * math.pi * end / 24),
            math.cos(2 * math.pi * end / 24),
        )


# ---------------------------------------------------------------------------
# Parser and validator
# ---------------------------------------------------------------------------

def parse_fingerprint(raw: dict) -> ModuleFingerprint:
    """
    Parse and validate a raw fingerprint dict (as provided by a module at
    registration time) into a ModuleFingerprint.

    Raises ValueError with a descriptive message on any validation failure.
    """
    _require_keys(raw, ["module_id", "cluster", "version",
                        "question_template", "default_threshold"])

    module_id = _require_str(raw, "module_id")
    cluster = _require_str(raw, "cluster")
    version = _require_str(raw, "version")
    question_template = _require_str(raw, "question_template")
    default_threshold = _require_float_in_range(raw, "default_threshold", 0.0, 1.0)

    priors = raw.get("signal_priors", {})
    if not isinstance(priors, dict):
        raise ValueError("signal_priors must be a dict")

    filesystem = _parse_filesystem_prior(priors.get("filesystem")) if "filesystem" in priors else None
    memory = _parse_memory_prior(priors.get("memory")) if "memory" in priors else None
    time_prior = _parse_time_prior(priors.get("time")) if "time" in priors else None

    # Validate question_template contains at least {location}
    if "{location}" not in question_template:
        raise ValueError(
            "question_template must contain '{location}' placeholder, got: "
            f"{question_template!r}"
        )

    return ModuleFingerprint(
        module_id=module_id,
        cluster=cluster,
        version=version,
        question_template=question_template,
        default_threshold=default_threshold,
        filesystem=filesystem,
        memory=memory,
        time=time_prior,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_keys(d: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Fingerprint missing required keys: {missing}")


def _require_str(d: dict, key: str) -> str:
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"'{key}' must be a non-empty string, got: {v!r}")
    return v.strip()


def _require_float_in_range(d: dict, key: str, lo: float, hi: float) -> float:
    v = d[key]
    if not isinstance(v, (int, float)):
        raise ValueError(f"'{key}' must be a number, got: {v!r}")
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(f"'{key}' must be in [{lo}, {hi}], got: {v}")
    return v


def _expand_dir(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _parse_filesystem_prior(raw: dict) -> FilesystemPrior:
    if not isinstance(raw, dict):
        raise ValueError("signal_priors.filesystem must be a dict")

    watch_dirs = raw.get("watch_directories", [])
    if not isinstance(watch_dirs, list) or not watch_dirs:
        raise ValueError(
            "signal_priors.filesystem.watch_directories must be a non-empty list"
        )
    watch_dirs = [_expand_dir(d) for d in watch_dirs]

    rel_exts = raw.get("relevant_extensions", [])
    irrel_exts = raw.get("irrelevant_extensions", [])
    if not isinstance(rel_exts, list):
        raise ValueError("signal_priors.filesystem.relevant_extensions must be a list")
    if not isinstance(irrel_exts, list):
        raise ValueError("signal_priors.filesystem.irrelevant_extensions must be a list")

    for ext in rel_exts + irrel_exts:
        if not isinstance(ext, str) or not ext.startswith("."):
            raise ValueError(
                f"Extensions must be strings starting with '.', got: {ext!r}"
            )

    return FilesystemPrior(
        watch_directories=watch_dirs,
        relevant_extensions=list(rel_exts),
        irrelevant_extensions=list(irrel_exts),
    )


def _parse_memory_prior(raw: dict) -> MemoryPrior:
    if not isinstance(raw, dict):
        raise ValueError("signal_priors.memory must be a dict")

    namespaces = raw.get("watch_namespaces", [])
    if not isinstance(namespaces, list) or not namespaces:
        raise ValueError(
            "signal_priors.memory.watch_namespaces must be a non-empty list"
        )
    for ns in namespaces:
        if not isinstance(ns, str):
            raise ValueError(f"watch_namespaces entries must be strings, got: {ns!r}")

    keys = raw.get("high_relevance_keys", [])
    if not isinstance(keys, list):
        raise ValueError("signal_priors.memory.high_relevance_keys must be a list")

    return MemoryPrior(
        watch_namespaces=list(namespaces),
        high_relevance_keys=list(keys),
    )


def _parse_time_prior(raw: dict) -> TimePrior:
    if not isinstance(raw, dict):
        raise ValueError("signal_priors.time must be a dict")

    active_hours = raw.get("active_hours")
    if (
        not isinstance(active_hours, list)
        or len(active_hours) != 2
        or not all(isinstance(h, int) and 0 <= h <= 23 for h in active_hours)
    ):
        raise ValueError(
            "signal_priors.time.active_hours must be a list of exactly two ints "
            "in [0, 23], got: {!r}".format(active_hours)
        )
    if active_hours[0] > active_hours[1]:
        raise ValueError(
            "signal_priors.time.active_hours start must be <= end, "
            "got: {!r}".format(active_hours)
        )

    active_days = raw.get("active_days", list(range(7)))
    if (
        not isinstance(active_days, list)
        or not all(isinstance(d, int) and 0 <= d <= 6 for d in active_days)
    ):
        raise ValueError(
            "signal_priors.time.active_days must be a list of ints in [0, 6], "
            "got: {!r}".format(active_days)
        )

    interval = raw.get("typical_interval_hours", 24.0)
    if not isinstance(interval, (int, float)) or float(interval) <= 0:
        raise ValueError(
            "signal_priors.time.typical_interval_hours must be a positive number, "
            "got: {!r}".format(interval)
        )

    return TimePrior(
        active_hours=(int(active_hours[0]), int(active_hours[1])),
        active_days=list(active_days),
        typical_interval_hours=float(interval),
    )
