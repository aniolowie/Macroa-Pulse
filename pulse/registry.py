# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
PulseRegistry , top-level coordinator of the Pulse subsystem.
Architecture: docs/ARCHITECTURE.md §Registry
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from pulse.bus import SignalBus
from pulse.fingerprint import ModuleFingerprint, parse_fingerprint
from pulse.limbic import LimbicLayer
from pulse.prefrontal import EscalationDecision, PrefrontalLayer
from pulse.retina import Retina, SignalEvent
from pulse.training import TrainingBuffer

# Delay before drain_training() is called after an activation is recorded.
_DRAIN_DELAY_SECONDS = 2.0
# Workers for per-module scoring (keeps _on_signal off the dispatcher thread).
_EXECUTOR_WORKERS = 4


class PulseRegistry:
    """
    Top-level coordinator owned by the kernel.

    Owns all Pulse layers. Does not start any background threads until start()
    is called.
    """

    def __init__(
        self,
        watch_dirs: list[str],
        threshold: float = 0.65,
        model_save_path: Optional[Path] = None,
        auto_save_interval: int = 10,
    ) -> None:
        self._watch_dirs = list(watch_dirs)
        self._model_save_path = model_save_path
        self._auto_save_interval = auto_save_interval
        self._activation_count = 0

        self._signal_queue: queue.Queue = queue.Queue()
        self._bus = SignalBus()
        self._retina = Retina(
            watch_dirs=self._watch_dirs,
            signal_queue=self._signal_queue,
        )
        self._limbic = LimbicLayer()
        self._prefrontal = PrefrontalLayer(threshold=threshold)
        self._training = TrainingBuffer()

        self._fingerprints: dict[str, ModuleFingerprint] = {}
        self._escalation_handler: Optional[Callable[[EscalationDecision], None]] = None
        self._executor = ThreadPoolExecutor(
            max_workers=_EXECUTOR_WORKERS,
            thread_name_prefix="pulse-scorer",
        )

    # ------------------------------------------------------------------
    # Module registration
    # ------------------------------------------------------------------

    def register_module(self, module_id: str, fingerprint_raw: dict) -> None:
        """Parse fingerprint, register with LimbicLayer, extend Retina watch list.

        Safe to call before or after start(). When called after start(), new
        directories are scheduled on the running observer immediately via
        Retina.add_watch_dir().
        """
        fingerprint = parse_fingerprint(fingerprint_raw)
        self._fingerprints[module_id] = fingerprint
        self._limbic.register(module_id, fingerprint)
        for d in fingerprint.watch_directories():
            self._retina.add_watch_dir(d)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start Retina and SignalBus; subscribe the internal signal handler."""
        self._bus.subscribe(self._on_signal)
        # Wire Retina output into the bus.
        _RetinaForwarder(self._signal_queue, self._bus).start()
        self._retina.start()

    def stop(self) -> None:
        """Stop Retina and SignalBus; persist model weights if path is set."""
        self._retina.stop()
        self._bus.stop()
        self._executor.shutdown(wait=False)
        if self._model_save_path is not None:
            self._limbic.save(self._model_save_path)

    # ------------------------------------------------------------------
    # Public API for the kernel
    # ------------------------------------------------------------------

    def on_escalation(self, handler: Callable[[EscalationDecision], None]) -> None:
        """Register (or replace) the escalation handler."""
        self._escalation_handler = handler

    def record_feedback(self, activation_id: str, label: float) -> None:
        """Delegate to TrainingBuffer."""
        self._training.record_feedback(activation_id, label)

    def record_activation(self, module_id: str, window: list[SignalEvent]) -> str:
        """Delegate to TrainingBuffer."""
        return self._training.record_activation(module_id, window)

    def drain_training(self) -> None:
        """Delegate to TrainingBuffer."""
        self._training.drain(self._limbic)

    # ------------------------------------------------------------------
    # Internal signal handling
    # ------------------------------------------------------------------

    def _on_signal(self, event: SignalEvent) -> None:
        """
        Called on the bus dispatcher thread for every SignalEvent.

        Offloads all per-module scoring to the thread pool immediately so the
        dispatcher thread is never blocked by LSTM inference.
        """
        module_ids = list(self._fingerprints.keys())
        for module_id in module_ids:
            self._executor.submit(self._score_and_evaluate, module_id, event)

    def _score_and_evaluate(self, module_id: str, event: SignalEvent) -> None:
        """
        Run in thread pool. Scores the event window for one module, evaluates
        escalation, and fires the handler if appropriate.
        """
        fingerprint = self._fingerprints.get(module_id)
        if fingerprint is None:
            return

        # A single event is treated as a window of length 1.
        window = [event]
        score = self._limbic.score(module_id, window)

        decision = self._prefrontal.evaluate(module_id, score, window, fingerprint)
        if not decision.should_escalate:
            return

        activation_id = self.record_activation(module_id, window)

        self._activation_count += 1
        if (
            self._activation_count % self._auto_save_interval == 0
            and self._model_save_path is not None
        ):
            self._limbic.save(self._model_save_path)

        handler = self._escalation_handler
        if handler is not None:
            try:
                handler(decision)
            except Exception:
                pass

        # Schedule drain after a short delay so the caller can attach feedback
        # before the buffer is flushed.
        t = threading.Timer(_DRAIN_DELAY_SECONDS, self.drain_training)
        t.daemon = True
        t.start()


# ---------------------------------------------------------------------------
# Internal helper: forwards events from Retina's queue to the SignalBus
# ---------------------------------------------------------------------------

class _RetinaForwarder:
    """
    Reads from the Retina's output queue and forwards each SignalEvent to the
    SignalBus. Runs on a single daemon thread.
    """

    def __init__(self, signal_queue: queue.Queue, bus: SignalBus) -> None:
        self._queue = signal_queue
        self._bus = bus
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="retina-forwarder",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=1.0)
                self._bus.put(event)
            except Exception:
                pass
