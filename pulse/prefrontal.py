# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Layer 3: Prefrontal , escalation gating and question formation.
Architecture: docs/ARCHITECTURE.md §Layer 3
"""

from __future__ import annotations

from dataclasses import dataclass

from pulse.fingerprint import ModuleFingerprint
from pulse.retina import SignalEvent


@dataclass
class EscalationDecision:
    module_id: str
    should_escalate: bool
    question: str | None
    confidence: float
    window: list[SignalEvent]


class PrefrontalLayer:
    """
    Layer 3: decides whether a high-scoring window warrants waking the agent,
    and if so, what to ask it.

    Invariant: should_escalate is True only when question is a non-empty string.
    """

    def __init__(self, threshold: float = 0.65) -> None:
        self._threshold = threshold

    def evaluate(
        self,
        module_id: str,
        score: float,
        window: list[SignalEvent],
        fingerprint: ModuleFingerprint,
    ) -> EscalationDecision:
        """
        Gate and form the escalation question.

        Returns should_escalate=False if:
          - score is below threshold
          - the question template is missing or empty
          - template substitution produces an empty string
          - template substitution raises any exception
        """
        def _no(confidence: float = score) -> EscalationDecision:
            return EscalationDecision(
                module_id=module_id,
                should_escalate=False,
                question=None,
                confidence=confidence,
                window=window,
            )

        if score < self._threshold:
            return _no()

        template = fingerprint.question_template
        if not template or not template.strip():
            return _no()

        location = window[-1].location if window else ""

        try:
            question = template.format(location=location)
        except (KeyError, ValueError, IndexError):
            return _no()

        if not question.strip():
            return _no()

        return EscalationDecision(
            module_id=module_id,
            should_escalate=True,
            question=question,
            confidence=score,
            window=window,
        )

    def set_threshold(self, t: float) -> None:
        """Update the escalation threshold at runtime."""
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {t!r}")
        self._threshold = t
