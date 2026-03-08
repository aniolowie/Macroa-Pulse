# Pulse: Architecture Guide
### Macroa , Build Document 1 of 5
**Status:** Pre-implementation specification. Read this before writing any code.

---

## What This Document Is

This is the implementation specification for the **Pulse** , the proactive cognition subsystem of Macroa. It is written for Claude Code. Every section is a direct implementation instruction. Do not deviate from the architecture described here without updating this document first.

The Pulse is built **before the kernel, before the SDK, before the shell.** It is the first thing built because it is what makes Macroa different from every other agent framework. Without the Pulse, Macroa is reactive. With the Pulse, it is proactive.

---

## Design Philosophy

> **AI is used only where no deterministic process can do the job. Everything else is infrastructure.**

The Pulse embodies this principle recursively: it achieves proactive behaviour using zero LLM calls in normal operation. Layer 1 is deterministic. Layer 2 is a tiny local neural network. Layer 3 uses string templates. The LLM is only called when all three layers cannot resolve ambiguity , which should be rare.

---

## What the Pulse Does

The Pulse answers one question continuously, at near-zero cost:

> *"Given the current state of the environment, is there anything that deserves the agent's attention right now?"*

It does this without knowing what "homework" or "taxes" or "assignments" means. It knows patterns in signals. When a pattern matches something that has historically preceded a relevant event, it fires a scoped question upward to the kernel. The kernel wakes the agent with that specific question. The agent answers it. Done.

This is fundamentally different from a cron job. A cron job asks: *"is it 3pm?"* The Pulse asks: *"does the current state of the world look like a state that has historically preceded something worth doing?"*

---

## The Three Layers

```
ENVIRONMENT (files, memory, network, time)
        │ continuous stream of raw events
        ▼
┌─────────────────────────────────────────┐
│  LAYER 1: RETINA                        │
│  Deterministic event detection          │
│  Cost: ~0 CPU, always running           │
│  Output: sparse SignalEvent stream      │
└──────────────────┬──────────────────────┘
                   │ only when delta detected
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 2: LIMBIC FILTER                 │
│  Per-module cluster neural networks     │
│  Cost: milliseconds on CPU              │
│  Output: RelevanceScore per cluster     │
└──────────────────┬──────────────────────┘
                   │ only when score > threshold
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 3: PREFRONTAL FILTER             │
│  Template-based question formation      │
│  Cost: ~0 (string interpolation)        │
│  Output: ScopedQuestion for the kernel  │
└──────────────────┬──────────────────────┘
                   │ only when question formed
                   ▼
              KERNEL SIGNAL BUS
              (wakes agent with scoped question)
```

---

## Layer 1: Retina

### Responsibility

Detect changes in the environment. Emit a `SignalEvent` when something changes. Do not interpret. Do not reason. Just detect.

### What It Watches

| Signal Source | What Is Detected | How |
|---|---|---|
| File system | New file, modified file, deleted file in monitored directories | `watchdog` library (Python), `inotify` on Linux |
| Memory namespaces | A fact was written or updated in a monitored namespace | Internal event hook on the memory driver |
| Time | Cyclical time features: hour of day (0–23), day of week (0–6), time since last agent activation | Computed on a 60-second tick |
| Network (optional, v2) | A monitored HTTP endpoint returned a different response hash | Polling with hash comparison |

### Output: `SignalEvent`

```python
@dataclass
class SignalEvent:
    source: str          # "filesystem" | "memory" | "time" | "network"
    location: str        # path, namespace, or endpoint
    delta_type: str      # "created" | "modified" | "deleted" | "tick"
    magnitude: float     # 0.0–1.0, normalised change size
    timestamp: float     # unix timestamp
    features: dict       # source-specific features (see below)
```

**File system features:**
```python
{
    "path": "/home/user/Downloads/hw3.pdf",
    "extension": ".pdf",
    "size_bytes": 204800,
    "directory_depth": 3,
    "filename_tokens": ["hw3"]  # split on non-alphanumeric
}
```

**Memory features:**
```python
{
    "namespace": "/mem/homework/",
    "key": "last_checked",
    "value_length": 12
}
```

**Time tick features:**
```python
{
    "hour_sin": 0.866,   # sin(2π * hour / 24) , cyclical encoding
    "hour_cos": 0.5,
    "dow_sin": 0.782,    # sin(2π * day_of_week / 7)
    "dow_cos": 0.623,
    "minutes_since_last_activation": 847
}
```

### Implementation Notes

- Use Python's `watchdog` library for filesystem events. Configure it to watch only directories declared in module fingerprints. Do not watch the entire filesystem.
- The time tick fires every 60 seconds regardless of other events. It is the Pulse's heartbeat.
- Layer 1 runs in its own thread. It puts `SignalEvent` objects onto a thread-safe queue that Layer 2 consumes.
- Layer 1 has no state. It does not remember previous events. It only emits.

---

## Layer 2: Limbic Filter

### Responsibility

For each registered module cluster, maintain a small neural network that takes a window of recent `SignalEvent` objects and outputs a relevance score (0.0–1.0). A score above the threshold triggers Layer 3.

### Architecture: Per-Cluster Model

Each module cluster has its own independent model. Models run in parallel. No model knows about other models.

**Model architecture:** A small LSTM or Temporal Convolutional Network (TCN).

- Input: a sliding window of the last N `SignalEvent` feature vectors, flattened and padded
- Hidden size: 32–64 units
- Output: a single float (relevance score, 0.0–1.0) via sigmoid activation
- Parameters: approximately 50,000–200,000 (well under 1M)
- Inference time: < 5ms on CPU

**Why LSTM/TCN over a transformer:** transformers require fixed-size attention windows and are expensive for continuous inference. LSTMs and TCNs are designed for streaming time-series data, run efficiently on CPU, and handle variable-length sequences naturally.

### Cold Start: Module Fingerprint

When a module registers with the Pulse, it provides a **signal fingerprint** , a JSON structure that describes what signals are associated with its relevance. This fingerprint is used to initialise the cluster model's weights with a meaningful prior, so it is not random on day one.

**Signal fingerprint format:**

```json
{
    "module_id": "homework-agent",
    "cluster": "academic",
    "version": "1.0",
    "signal_priors": {
        "filesystem": {
            "watch_directories": ["~/Downloads", "~/Documents"],
            "relevant_extensions": [".pdf", ".docx", ".pptx"],
            "irrelevant_extensions": [".exe", ".zip", ".mp3"]
        },
        "memory": {
            "watch_namespaces": ["/mem/homework/", "/mem/courses/"],
            "high_relevance_keys": ["last_assignment", "due_date"]
        },
        "time": {
            "active_hours": [8, 22],
            "active_days": [0, 1, 2, 3, 4],
            "typical_interval_hours": 24
        }
    },
    "question_template": "A new file appeared at {location}. Is this file related to a course assignment or homework?",
    "default_threshold": 0.65
}
```

**Initialisation process:** The fingerprint is converted into synthetic training examples (positive and negative) that are used to pre-train the cluster model before any real data exists. This gives the model a reasonable prior on day one.

### Cluster Assignment

Modules are assigned to clusters based on their declared `cluster` field in the fingerprint. Multiple modules can share a cluster (e.g., "homework-agent" and "notes-agent" both belong to "academic"). When two modules share a cluster, they share a cluster model , the model fires for the cluster, and Layer 3 determines which specific module is relevant.

### Output: `RelevanceScore`

```python
@dataclass
class RelevanceScore:
    cluster_id: str
    score: float          # 0.0–1.0
    triggering_events: list[SignalEvent]  # the events that contributed
    timestamp: float
```

### Training Loop

The training loop runs in a background thread. It updates cluster models using online learning (one gradient step per new labeled example). Labels come from two sources:

1. **Implicit:** if the agent was activated and took an action (wrote memory, ran a tool), the activation is labeled positive. If the agent was activated and did nothing, it is labeled negative.
2. **Explicit:** the shell can present a "was this useful?" prompt to the user. User response overrides implicit label.

Training data is stored locally in `~/.macroa/pulse/training_data.db` (SQLite). Models are stored in `~/.macroa/pulse/models/`. No data ever leaves the machine.

---

## Layer 3: Prefrontal Filter

### Responsibility

Take a `RelevanceScore` above threshold and produce a `ScopedQuestion` , a specific, focused question for the agent. The agent wakes up with this question and does not need to search through all modules.

### How It Works

1. Look up the cluster's registered modules and their question templates.
2. Identify which module's fingerprint best matches the triggering events (simple rule-based matching , no LLM).
3. Interpolate the question template with the specific signal details.
4. Emit a `ScopedQuestion` to the kernel signal bus.

**Example:**

- Triggering event: new file `/home/user/Downloads/hw3.pdf` created
- Cluster: "academic", best matching module: "homework-agent"
- Template: `"A new file appeared at {location}. Is this file related to a course assignment or homework?"`
- Result: `"A new file appeared at /home/user/Downloads/hw3.pdf. Is this file related to a course assignment or homework?"`

### Fallback: Ambiguous Cases

If Layer 3 cannot identify a single best-matching module (two modules in the cluster both match equally well), it forms a broader question:

`"Something changed in the {cluster} cluster. Specifically: {event_summary}. Which of the following is relevant: {module_list}?"`

This broader question is more expensive for the agent to answer, but it only occurs when Layer 3 genuinely cannot resolve the ambiguity deterministically.

### Output: `ScopedQuestion`

```python
@dataclass
class ScopedQuestion:
    question: str
    cluster_id: str
    module_id: str | None     # None if ambiguous
    triggering_events: list[SignalEvent]
    confidence: float         # Layer 2 score that triggered this
    timestamp: float
```

---

## File Structure

```
pulse/
├── __init__.py
├── retina.py          # Layer 1: event detection, SignalEvent dataclass
├── limbic.py          # Layer 2: cluster models, training loop, RelevanceScore
├── prefrontal.py      # Layer 3: question formation, ScopedQuestion
├── fingerprint.py     # Fingerprint parsing, validation, synthetic prior generation
├── bus.py             # Internal event queue between layers
├── registry.py        # Module registration, cluster assignment
├── training.py        # Online learning loop, label collection
└── models/            # Saved cluster model weights (PyTorch .pt files)
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.0 | LSTM/TCN models, training loop |
| `watchdog` | >= 3.0 | Filesystem event detection |
| `numpy` | >= 1.24 | Feature vector construction |
| `sqlite3` | stdlib | Training data storage |

No LLM API calls. No network requests. No external services.

---

## Build Order Within Pulse

Build in this exact order. Do not skip ahead.

1. `retina.py` , implement `SignalEvent`, filesystem watcher, time tick. Test with a simple script that prints events.
2. `fingerprint.py` , implement fingerprint parsing and validation. Test with the example fingerprint above.
3. `bus.py` , implement the thread-safe queue between layers.
4. `limbic.py` , implement the LSTM model, cluster registry, and inference. Test with synthetic data from the fingerprint.
5. `training.py` , implement the online learning loop and SQLite storage.
6. `prefrontal.py` , implement template interpolation and question formation.
7. `registry.py` , implement module registration and cluster assignment.
8. `__init__.py` , wire all layers together and expose `Pulse.start()`, `Pulse.register_module()`, `Pulse.subscribe()`.

---

## Integration Contract (for the Kernel, built later)

The Pulse exposes three public methods:

```python
class Pulse:
    def start(self) -> None:
        """Start all three layers. Non-blocking. Runs in background threads."""

    def register_module(self, fingerprint: dict) -> None:
        """Register a module with the Pulse. Initialises its cluster model."""

    def subscribe(self, callback: Callable[[ScopedQuestion], None]) -> None:
        """Register a callback that fires when a ScopedQuestion is produced."""

    def label_activation(self, question: ScopedQuestion, useful: bool) -> None:
        """Provide a training label for a past activation. Called by the kernel."""

    def stop(self) -> None:
        """Gracefully stop all layers and save model state."""
```

The kernel calls `subscribe()` with a callback that wakes the agent. The kernel calls `label_activation()` after the agent finishes to provide the training signal. That is the entire interface. The Pulse does not know about the kernel's internals.
