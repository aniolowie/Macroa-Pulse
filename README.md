# Macroa-Pulse

**Proactive cognition for AI agents — without cron jobs, webhooks, or LLM polling.**

[![Documentation](https://img.shields.io/badge/docs-pulse.macroa.org-6366F1?style=flat-square)](https://pulse.macroa.org/docs)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://www.python.org/)
[![Status: Proof of Concept](https://img.shields.io/badge/status-proof%20of%20concept-orange?style=flat-square)](https://pulse.macroa.org/docs)

Macroa-Pulse is the proactive cognition subsystem of [Macroa](https://github.com/aniolowie/Macroa), an AI agent operating system. It enables AI agents to notice when something is worth attention and act on it — without being triggered by a user, a scheduler, or an external event.

**[Full documentation at pulse.macroa.org/docs](https://pulse.macroa.org/docs)**

## The Problem

Every AI agent framework today is reactive. An agent acts because something triggered it. Remove the trigger, and the agent does nothing. Building a genuinely useful AI assistant requires the developer to anticipate every situation in which the agent should act and write an explicit trigger for it. The Pulse eliminates this requirement.

## How It Works

The Pulse is a three-layer hierarchical signal perception system inspired by the brain's perceptual architecture:

```
ENVIRONMENT (files, memory, network, time)
        │
        ▼
  LAYER 1: RETINA          — deterministic change detection, ~0 cost
        │
        ▼
  LAYER 2: LIMBIC FILTER   — tiny per-cluster neural networks (LSTM/TCN), <5ms on CPU
        │
        ▼
  LAYER 3: PREFRONTAL      — template-based question formation, ~0 cost
        │
        ▼
  AGENT (wakes with a focused, scoped question)
```

In normal operation, the Pulse runs entirely on local compute with **zero LLM API calls**. The agent is only invoked when all three layers agree something is worth attention — and it wakes up with a specific question, not a blank slate.

## Quickstart

### Install

```bash
# From source (PyPI release coming soon)
git clone https://github.com/aniolowie/Macroa-Pulse.git
cd Macroa-Pulse
pip install -e .
```

### Monitor a directory and react proactively

```python
import time
from pulse import PulseRegistry, EscalationDecision

registry = PulseRegistry(watch_dirs=["~/my-folder"], threshold=0.65)

registry.register_module("my-agent", {
    "module_id": "my-agent",
    "cluster": "documents",
    "version": "1.0",
    "question_template": "A new file appeared at {location}. Is this relevant?",
    "default_threshold": 0.65,
    "signal_priors": {
        "filesystem": {
            "watch_directories": ["~/my-folder"],
            "relevant_extensions": [".pdf", ".txt", ".docx"],
            "irrelevant_extensions": [".tmp", ".log"]
        }
    }
})

def on_escalation(decision: EscalationDecision):
    print(f"Agent question: {decision.question}")
    print(f"Confidence:     {decision.confidence:.2f}")

registry.on_escalation(on_escalation)
registry.start()

print("Pulse running. Drop a file into ~/my-folder to trigger it.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    registry.stop()
```

### Or use the CLI

```bash
# Watch a directory with a fingerprint file and an LLM provider
macroa-pulse --watch ~/my-folder --module fingerprint.json --provider anthropic
```

See the [full documentation](https://pulse.macroa.org/docs) for signal fingerprints, CLI reference, architecture deep-dives, and integration guides.

## Status

**Active development — Proof of Concept**

The Pulse is the first component of Macroa to be built.

## Documentation

Full docs: **[pulse.macroa.org/docs](https://pulse.macroa.org/docs)**

- [Introduction](https://pulse.macroa.org/docs) — what Pulse is and why it exists
- [Quickstart](https://pulse.macroa.org/docs/quickstart) — get running in minutes
- [Architecture](https://pulse.macroa.org/docs/concepts/three-layer-architecture) — how the three layers work
- [API Reference](https://pulse.macroa.org/docs/api/pulse-registry) — full API surface

## Part of Macroa

Macroa-Pulse is the first module of the [Macroa](https://github.com/aniolowie/Macroa) platform — an AI agent operating system designed around the principle that AI should be used only where no deterministic process can do the job.

## License

[AGPLv3](LICENSE) — free for everyone. Modifications must be published under the same license.
