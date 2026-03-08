# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Macroa-Pulse: Proactive cognition subsystem for AI agents.

Architecture: docs/ARCHITECTURE.md
"""

from pulse.fingerprint import ModuleFingerprint
from pulse.prefrontal import EscalationDecision
from pulse.registry import PulseRegistry
from pulse.retina import SignalEvent

__all__ = [
    "EscalationDecision",
    "ModuleFingerprint",
    "PulseRegistry",
    "SignalEvent",
]
