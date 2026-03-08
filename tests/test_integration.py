#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Integration test: PulseRegistry detects a new .txt file and escalates.

Run with:  python3 tests/test_integration.py
"""

import os
import shutil
import sys
import tempfile
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pulse import EscalationDecision, PulseRegistry


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="pulse_test_")
    print(f"Temp directory: {tmp_dir}")

    decisions: list[EscalationDecision] = []

    registry = PulseRegistry(watch_dirs=[tmp_dir], threshold=0.0)

    registry.register_module(
        "test",
        {
            "module_id": "test",
            "cluster": "test-cluster",
            "version": "0.1",
            "question_template": "Is {location} relevant?",
            "default_threshold": 0.0,
            "signal_priors": {
                "filesystem": {
                    "watch_directories": [tmp_dir],
                    "relevant_extensions": [".txt"],
                    "irrelevant_extensions": [],
                }
            },
        },
    )

    registry.on_escalation(decisions.append)
    registry.start()

    print("Waiting 1 s for Retina to initialise...")
    time.sleep(1)

    test_file = os.path.join(tmp_dir, "hello.txt")
    print(f"Creating {test_file}")
    with open(test_file, "w") as f:
        f.write("test content")

    print("Waiting 3 s for signal to propagate...")
    time.sleep(3)

    registry.stop()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Temp directory removed. Decisions received: {len(decisions)}")

    # --- assertions ---
    assert len(decisions) >= 1, (
        f"Expected at least one EscalationDecision, got {len(decisions)}"
    )
    escalated = [d for d in decisions if d.should_escalate and d.module_id == "test"]
    assert escalated, (
        f"No decision had should_escalate=True and module_id='test'. "
        f"Decisions: {decisions}"
    )

    print(f"\nPASS — received {len(escalated)} escalation(s) for module 'test'")
    for d in escalated:
        print(f"  question={d.question!r}  confidence={d.confidence:.4f}")


if __name__ == "__main__":
    main()
