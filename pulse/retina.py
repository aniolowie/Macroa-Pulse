# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Layer 1: Retina , deterministic change detection.
Architecture: docs/ARCHITECTURE.md §Layer 1
"""

import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import zlib

import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

FEATURE_DIM = 16

_DELTA_TYPE_ENC = {"created": 1.0, "modified": 0.5, "deleted": 0.0, "tick": 0.75}
_SOURCE_ENC = {"filesystem": 1.0, "time": 0.5, "memory": 0.25, "network": 0.0}


@dataclass
class SignalEvent:
    source: str       # "filesystem" | "memory" | "time" | "network"
    location: str     # path, namespace, or endpoint
    delta_type: str   # "created" | "modified" | "deleted" | "tick"
    magnitude: float  # 0.0–1.0, normalised change size
    timestamp: float  # unix timestamp
    features: dict    # source-specific features

    def to_feature_vector(self) -> np.ndarray:
        is_fs = self.source == "filesystem"
        is_time = self.source == "time"

        v = [
            float(self.magnitude),                                                          # [0]
            float(_DELTA_TYPE_ENC.get(self.delta_type, 0.0)),                              # [1]
            float(_SOURCE_ENC.get(self.source, 0.0)),                                      # [2]
            float(self.features.get("hour_sin", 0.0)) if is_time else 0.0,                 # [3]
            float(self.features.get("hour_cos", 0.0)) if is_time else 0.0,                 # [4]
            float(self.features.get("dow_sin", 0.0)) if is_time else 0.0,                  # [5]
            float(self.features.get("dow_cos", 0.0)) if is_time else 0.0,                  # [6]
            float(self.features.get("minutes_since_last_activation", 0.0)) if is_time else 0.0,  # [7]
            _normalise_size(self.features.get("size_bytes", 0)) if is_fs else 0.0,         # [8]
            min(float(self.features.get("directory_depth", 0)) / 10.0, 1.0) if is_fs else 0.0,  # [9]
            (zlib.crc32(self.features.get("extension", "").encode()) % 1000) / 1000.0 if is_fs else 0.0,  # [10]
            0.0, 0.0, 0.0, 0.0, 0.0,                                                       # [11–15] reserved
        ]

        result = np.array(v, dtype=np.float32)
        assert len(result) == FEATURE_DIM
        return result


# Reference ceiling for log-normalising file sizes (1 GiB).
_MAX_SIZE_BYTES = 1024 * 1024 * 1024


def _normalise_size(size_bytes: int) -> float:
    if size_bytes <= 0:
        return 0.0
    return min(math.log1p(size_bytes) / math.log1p(_MAX_SIZE_BYTES), 1.0)


def _filename_tokens(stem: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", stem) if t]


def _fs_features(path: str) -> dict:
    p = Path(path)
    try:
        size_bytes = p.stat().st_size
    except OSError:
        size_bytes = 0
    return {
        "path": path,
        "extension": p.suffix,
        "size_bytes": size_bytes,
        "directory_depth": len(p.parts) - 1,
        "filename_tokens": _filename_tokens(p.stem),
    }


class _FSHandler(FileSystemEventHandler):
    def __init__(self, signal_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = signal_queue

    def _emit(self, path: str, delta_type: str) -> None:
        features = _fs_features(path)
        if delta_type == "modified":
            magnitude = _normalise_size(features["size_bytes"])
        else:
            magnitude = 1.0
        self._queue.put(SignalEvent(
            source="filesystem",
            location=path,
            delta_type=delta_type,
            magnitude=magnitude,
            timestamp=time.time(),
            features=features,
        ))

    def on_created(self, event):
        if not event.is_directory:
            self._emit(event.src_path, "created")

    def on_modified(self, event):
        if not event.is_directory:
            self._emit(event.src_path, "modified")

    def on_deleted(self, event):
        if not event.is_directory:
            self._emit(event.src_path, "deleted")


class Retina:
    """
    Layer 1: deterministic event detection. No inter-event state; only emits.

    Watches declared directories for filesystem events via watchdog and fires a
    60-second time tick with cyclically-encoded temporal features. All
    SignalEvent objects are placed on the caller-supplied thread-safe queue.

    Args:
        watch_dirs: Directories to monitor. Only these paths are watched, not
            the entire filesystem. Non-existent directories are silently skipped.
        signal_queue: The thread-safe queue that receives SignalEvent objects.
        get_minutes_since_activation: Optional callable that returns the number
            of minutes elapsed since the last agent activation. When omitted,
            minutes are measured from Retina.start().
    """

    TICK_INTERVAL: int = 60  # seconds

    def __init__(
        self,
        watch_dirs: list[str],
        signal_queue: queue.Queue,
        get_minutes_since_activation: Optional[Callable[[], float]] = None,
    ) -> None:
        self._watch_dirs = [
            str(Path(d).expanduser().resolve()) for d in watch_dirs
        ]
        self._watched_set: set[str] = set()
        self._queue = signal_queue
        self._get_minutes_since_activation = get_minutes_since_activation
        self._start_time: Optional[float] = None
        self._observer: Optional[Observer] = None
        self._handler: Optional[_FSHandler] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start filesystem observer and time-tick thread. Non-blocking."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._start_observer()
        self._start_tick_thread()

    def stop(self) -> None:
        """Gracefully stop all background threads."""
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def add_watch_dir(self, path: str) -> None:
        """
        Add a directory to the watchdog observer at runtime.

        Safe to call before or after start(). Silently skips paths that are
        already watched or do not exist on disk.
        """
        resolved = str(Path(path).expanduser().resolve())
        if resolved in self._watched_set:
            return
        self._watch_dirs.append(resolved)
        if self._observer is not None and self._handler is not None:
            if os.path.isdir(resolved):
                self._observer.schedule(self._handler, resolved, recursive=True)
                self._watched_set.add(resolved)

    def _start_observer(self) -> None:
        self._handler = _FSHandler(self._queue)
        self._observer = Observer()
        for d in self._watch_dirs:
            if os.path.isdir(d):
                self._observer.schedule(self._handler, d, recursive=True)
                self._watched_set.add(d)
        self._observer.start()

    def _start_tick_thread(self) -> None:
        t = threading.Thread(
            target=self._tick_loop,
            name="retina-tick",
            daemon=True,
        )
        t.start()

    def _tick_loop(self) -> None:
        while not self._stop_event.wait(self.TICK_INTERVAL):
            self._emit_tick()

    def _emit_tick(self) -> None:
        now = time.time()
        lt = time.localtime(now)
        hour = lt.tm_hour
        dow = lt.tm_wday  # 0 = Monday, matching Python convention

        if self._get_minutes_since_activation is not None:
            minutes_since = self._get_minutes_since_activation()
        else:
            minutes_since = (now - self._start_time) / 60.0

        features = {
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "dow_sin": math.sin(2 * math.pi * dow / 7),
            "dow_cos": math.cos(2 * math.pi * dow / 7),
            "minutes_since_last_activation": round(minutes_since, 2),
        }

        self._queue.put(SignalEvent(
            source="time",
            location="tick",
            delta_type="tick",
            magnitude=1.0,
            timestamp=now,
            features=features,
        ))
