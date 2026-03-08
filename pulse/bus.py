# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
SignalBus , internal event queue between Layer 1 (Retina) and Layer 2 (Limbic).
See docs/ARCHITECTURE.md §Layer 1 / §Layer 2.

Architecture
------------
Retina calls put() on one thread. Layer 2 may consume events in two ways:

  Pull: call get(timeout) to block until the next event arrives.
  Push: call subscribe(callback) to receive events on a dedicated dispatcher
        thread as soon as they are produced.

Both modes are supported simultaneously. Every event produced by put() is
delivered to ALL registered subscribers AND made available via get(), with no
event being dropped or split between modes. This fan-out is performed by a
single internal dispatcher thread.

                         ┌──────────────┐
    Retina               │  _raw_queue  │
    put(event) ─────────►│  queue.Queue │
                         └──────┬───────┘
                                │  dispatcher thread
                    ┌───────────┼────────────┐
                    ▼           ▼            ▼
             _pull_queue   callback[0]  callback[1] …
             (get() reads)
"""

import queue
import threading
from typing import Callable, Optional

from pulse.retina import SignalEvent

# Sentinel placed on _raw_queue to signal the dispatcher thread to exit.
_STOP = object()


class SignalBus:
    """
    Thread-safe event bus connecting Retina (producer) to Limbic (consumer).

    The bus starts its dispatcher thread immediately on construction and runs
    until stop() is called. All methods are safe to call from any thread.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """
        Args:
            maxsize: Maximum number of events buffered in the raw queue before
                put() blocks. 0 (default) means unbounded.
        """
        self._raw_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._pull_queue: queue.Queue = queue.Queue()
        self._subscribers: list[Callable[[SignalEvent], None]] = []
        self._lock = threading.Lock()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="signal-bus-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def put(self, event: SignalEvent) -> None:
        """
        Enqueue a SignalEvent. Called by Retina. Non-blocking unless the bus
        was constructed with a finite maxsize and the queue is full, in which
        case it blocks until space is available.
        """
        self._raw_queue.put(event)

    # ------------------------------------------------------------------
    # Consumer API , pull mode
    # ------------------------------------------------------------------

    def get(self, timeout: float = 1.0) -> Optional[SignalEvent]:
        """
        Block until the next event is available and return it, or return None
        if timeout expires first. Does not consume events that have already
        been delivered to subscribers.

        Args:
            timeout: Seconds to wait. Must be >= 0.
        """
        try:
            return self._pull_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Consumer API , push mode
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[SignalEvent], None]) -> None:
        """
        Register a callback that will be called on the dispatcher thread for
        every subsequent event. Callbacks must not block for extended periods;
        offload heavy work to a separate thread if needed.

        Callbacks registered after some events have already been dispatched
        will only receive future events.
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[SignalEvent], None]) -> None:
        """Remove a previously registered callback. No-op if not registered."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """
        Signal the dispatcher thread to exit and wait for it to finish.
        After stop() returns, no further callbacks will be invoked and get()
        will drain any remaining events already in the pull queue.
        """
        self._raw_queue.put(_STOP)
        self._dispatcher.join()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while True:
            item = self._raw_queue.get()
            if item is _STOP:
                break
            event: SignalEvent = item
            # Fan out to pull queue first so get() callers are unblocked.
            self._pull_queue.put(event)
            # Snapshot subscribers under the lock, then call outside the lock
            # so that subscribe()/unsubscribe() cannot deadlock against a
            # slow callback.
            with self._lock:
                callbacks = list(self._subscribers)
            for cb in callbacks:
                try:
                    cb(event)
                except Exception:
                    # A misbehaving subscriber must not kill the dispatcher.
                    pass
