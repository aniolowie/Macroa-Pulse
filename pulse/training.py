# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Feedback loop: records activations and trains LimbicLayer models online.
Architecture: docs/ARCHITECTURE.md §Training
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from pulse.limbic import LimbicLayer
from pulse.retina import SignalEvent

_FEEDBACK_TIMEOUT = 300.0  # 5 minutes in seconds
_IMPLICIT_LABEL_WITH_OUTPUT = 0.8
_IMPLICIT_LABEL_NO_OUTPUT = 0.2


@dataclass
class ActivationRecord:
    module_id: str
    window: list[SignalEvent]
    timestamp: float
    label: float | None = field(default=None)


class TrainingBuffer:
    """
    Stores ActivationRecords and drives online weight updates for LimbicLayer.

    Lifecycle of a record:
      1. record_activation() stores it with label=None.
      2. record_feedback() sets the label when explicit feedback arrives.
      3. drain() trains the model for all labelled records (and for unlabelled
         records older than 5 minutes, using infer_label()), then removes them.
    """

    def __init__(self) -> None:
        self._records: dict[str, ActivationRecord] = {}

    def record_activation(
        self,
        module_id: str,
        window: list[SignalEvent],
    ) -> str:
        """Store an ActivationRecord and return its unique activation_id."""
        activation_id = uuid.uuid4().hex
        self._records[activation_id] = ActivationRecord(
            module_id=module_id,
            window=window,
            timestamp=time.time(),
        )
        return activation_id

    def record_feedback(self, activation_id: str, label: float) -> None:
        """
        Attach an explicit label to a stored record.

        label must be in [0.0, 1.0]. Silently ignores unknown activation_ids.
        """
        if activation_id not in self._records:
            return
        if not 0.0 <= label <= 1.0:
            raise ValueError(f"label must be in [0.0, 1.0], got {label!r}")
        self._records[activation_id].label = label

    def drain(self, limbic: LimbicLayer) -> None:
        """
        Train on all ready records, then remove them from the buffer.

        A record is ready if it has an explicit label, or if it is older than
        5 minutes (in which case infer_label() is used as a fallback).

        Records with no label and younger than 5 minutes are kept.
        """
        now = time.time()
        ids_to_remove: list[str] = []

        for activation_id, record in self._records.items():
            if record.label is not None:
                label = record.label
            elif now - record.timestamp >= _FEEDBACK_TIMEOUT:
                label = self.infer_label(record)
            else:
                continue  # still waiting for explicit feedback

            limbic.update_weights(record.module_id, record.window, label)
            ids_to_remove.append(activation_id)

        for activation_id in ids_to_remove:
            del self._records[activation_id]

    @staticmethod
    def infer_label(record: ActivationRecord) -> float:
        """
        Implicit feedback heuristic used when no explicit label arrives within
        the feedback timeout window.

        Returns 0.5 (neutral) because the window is always non-empty (it is
        what triggered the activation), making the old 0.8/0.2 heuristic
        always wrong. Neutral labels prevent corrupting the model before the
        kernel exists to supply real outcome metadata.
        """
        return 0.5
