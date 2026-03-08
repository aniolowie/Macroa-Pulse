# Macroa-Pulse

**Proactive cognition for AI agents — without cron jobs, webhooks, or LLM polling.**

Macroa-Pulse is the proactive cognition subsystem of [Macroa](https://github.com/aniolowie/Macroa), an AI agent operating system. It enables AI agents to notice when something is worth attention and act on it — without being triggered by a user, a scheduler, or an external event.

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

## Status

🔬 **Active development — Proof of Concept**

The Pulse is the first component of Macroa to be built. See the [architecture guide](docs/ARCHITECTURE.md) and the [design paper](docs/WHITEPAPER.md).

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) — implementation specification
- [Design Paper](docs/WHITEPAPER.md) — the problem, the approach, and the design rationale

## Part of Macroa

Macroa-Pulse is the first module of the [Macroa](https://github.com/aniolowie/Macroa) platform — an AI agent operating system designed around the principle that AI should be used only where no deterministic process can do the job.

## License

[AGPLv3](LICENSE) — free for everyone. Modifications must be published under the same license.
