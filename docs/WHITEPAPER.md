# The Pulse Architecture: Proactive Cognition in AI Agent Systems via Hierarchical Signal Perception

**Author:** Maciej (aniolowie)
**Project:** Macroa — An AI Agent Operating System
**Repository:** https://github.com/aniolowie/Macroa
**Status:** Proof of Concept / Design Specification

---

## Abstract

Every AI agent system built today is fundamentally reactive. An agent acts because something triggered it — a user message, a cron job, a webhook, an API call. Remove the trigger, and the agent does nothing. This paper describes the **Pulse**, a hierarchical signal perception architecture that enables AI agents to act proactively, without external triggers, at near-zero computational cost. The Pulse draws on principles from neuroscience — specifically the brain's hierarchical perceptual processing and default mode network — to create a three-layer system that continuously monitors the environment, learns behavioural patterns from usage, and produces scoped, focused questions for the agent only when the environment suggests something is worth attention. The architecture is designed to run entirely locally, requires no LLM calls in normal operation, and improves in accuracy over time through online learning from implicit and explicit user feedback.

---

## 1. The Problem: Reactive Agents and the Cost of Proactivity

### 1.1 All Current Agent Systems Are Reactive

The dominant paradigm in AI agent frameworks — LangChain, AutoGen, CrewAI, OpenAI Agents, and others — is reactive. An agent is a function: given an input, produce an output. The input is always provided by something external: a user, a scheduler, a webhook, or another agent. The agent has no internal drive to act.

This is not a limitation of the underlying models. It is a limitation of the architecture. The models themselves are capable of complex reasoning and planning. But they are never asked to reason about *whether* to act — only *how* to act, given that they have already been told to.

The consequence is that building a genuinely useful AI assistant requires the developer to anticipate every situation in which the agent should act and write an explicit trigger for it. A homework agent needs a cron job. A notification agent needs a webhook. A monitoring agent needs a polling loop. The developer is doing the cognitive work that the agent should be doing.

### 1.2 The Naive Solution and Why It Fails

The obvious solution is to run the agent continuously, polling for relevant events. Ask the LLM every few seconds: "is there anything I should be doing right now?" This would work. It would also cost approximately $10 per minute at current API prices for capable models, making it economically absurd for any real-world deployment.

The cost of LLM inference is the fundamental constraint. A model capable of reasoning about whether something is worth doing is the same model that costs money to run. You cannot afford to run it continuously. But if you do not run it continuously, you cannot achieve proactivity.

### 1.3 The Biological Insight

The human brain solves this problem. A person does not think about their homework every second of every day. But when they see a familiar file, or a classmate mentions an assignment, something in their brain fires: *"I should check on that."* This happens before conscious reasoning. It is fast, cheap, and usually accurate.

The brain achieves this through hierarchical perceptual processing. Raw sensory input is processed by cheap, fast, low-level systems that detect change and pattern. Only when these low-level systems flag something as potentially relevant does the expensive, slow, high-level reasoning system engage. The expensive system (the prefrontal cortex) never sees raw sensory data — it only sees pre-filtered, pre-interpreted signals from the layers below it.

The Pulse applies this principle to AI agent systems.

---

## 2. The Pulse Architecture

### 2.1 Overview

The Pulse is a three-layer hierarchical signal perception system that runs as a background process alongside the AI agent. It continuously monitors the environment, learns patterns from usage, and emits **scoped questions** to the agent when the environment suggests something is worth attention.

```
ENVIRONMENT (files, memory, network, time)
        │
        ▼
┌─────────────────────────────────────────┐
│  LAYER 1: RETINA                        │
│  Deterministic change detection         │
│  Cost: ~0                               │
└──────────────────┬──────────────────────┘
                   │ sparse SignalEvents
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 2: LIMBIC FILTER                 │
│  Tiny neural networks, one per cluster  │
│  Cost: milliseconds on CPU              │
└──────────────────┬──────────────────────┘
                   │ RelevanceScore (when > threshold)
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 3: PREFRONTAL FILTER             │
│  Template-based question formation      │
│  Cost: ~0 (string interpolation)        │
└──────────────────┬──────────────────────┘
                   │ ScopedQuestion
                   ▼
              AI AGENT (wakes with focused question)
```

Each layer acts as a gate. The vast majority of environmental changes are filtered out by Layer 1 (no delta detected) or Layer 2 (low relevance score). The agent is only woken when all three layers agree that something is worth attention.

### 2.2 Layer 1: The Retina

The Retina is a deterministic event detector. It watches four signal sources: file system events (new, modified, or deleted files in monitored directories), memory namespace deltas (facts written to the agent's persistent memory store), time signals (cyclically-encoded hour, day of week, and elapsed time since last activation), and optionally network events (hash comparison on monitored HTTP endpoints).

The Retina has no state and no intelligence. It only detects and emits `SignalEvent` objects — structured records containing the source, location, type of change, and source-specific features. Time features are cyclically encoded using sine/cosine transforms to correctly represent the circular nature of time (23:00 and 01:00 are close, not far apart in a cyclical encoding).

### 2.3 Layer 2: The Limbic Filter

The Limbic Filter is the core learning component of the Pulse. It maintains a small neural network for each registered **module cluster** — a group of modules that tend to be relevant in similar contexts. Each cluster model is a small LSTM (Long Short-Term Memory) network or Temporal Convolutional Network, taking a sliding window of recent `SignalEvent` feature vectors as input and producing a single relevance score (0.0–1.0) as output.

The model has approximately 50,000–200,000 parameters — small enough to run on CPU in under 5 milliseconds. This is not a limitation but a deliberate design choice: the Limbic Filter is not trying to understand what the signals mean. It is trying to recognise patterns in numbers. A small LSTM is the correct tool for this task.

**Cold start via module fingerprints:** The critical challenge for any personalised prediction system is the cold-start problem — the model is useless before it has seen any data. The Pulse solves this through **module fingerprints**: structured JSON documents that each module provides at registration time, describing what signals are associated with its relevance. These fingerprints are converted into synthetic training examples that pre-initialise the cluster model's weights before any real data exists. The model starts with a reasonable prior on day one and refines it from real usage.

**Online learning:** After each agent activation, the kernel provides a training label (positive if the agent found something useful and took action; negative if the agent did nothing). The cluster model is updated with a single gradient step. Over time, the model becomes increasingly accurate at predicting when its cluster is relevant.

### 2.4 Layer 3: The Prefrontal Filter

The Prefrontal Filter converts a relevance signal into a specific, actionable question for the agent. This is the critical step that prevents the agent from waking up with a blank slate and needing to search through all installed modules.

The filter works through template interpolation. Each module provides a `question_template` in its fingerprint — a string with placeholders for specific signal details. When Layer 2 fires, Layer 3 identifies the best-matching module within the cluster using simple rule-based matching and interpolates the template with the actual signal details.

For example: a new file `/home/user/Downloads/hw3.pdf` triggers the "academic" cluster. Layer 3 identifies the `homework-agent` as the best match and produces: *"A new file appeared at /home/user/Downloads/hw3.pdf. Is this file related to a course assignment?"* The agent wakes up with this specific question, not a blank slate.

---

## 3. Key Properties

### 3.1 Zero LLM Cost in Normal Operation

The Pulse runs entirely on local compute in normal operation. Layer 1 is deterministic. Layer 2 is a tiny neural network running on CPU. Layer 3 is string interpolation. No API calls are made. The cost of running the Pulse continuously is electricity, not tokens.

### 3.2 Improving Accuracy Over Time

Unlike a cron job, which is equally accurate (or inaccurate) forever, the Pulse improves with usage. Each agent activation provides a training signal. Over time, the cluster models learn the user's specific patterns: when their homework tends to appear, what their relevant file types look like, how their usage patterns vary by day of week.

### 3.3 Differentiation Between Similar Events

Modules that are genuinely similar are placed in the same cluster and share a cluster model. The cluster model fires for the cluster as a whole. Layer 3 then performs finer-grained differentiation using the module fingerprints. This separation of concerns — cluster-level pattern recognition in Layer 2, module-level specificity in Layer 3 — allows the system to handle similar events correctly.

### 3.4 Privacy by Design

All data — training examples, model weights, signal history — is stored locally on the user's machine. Nothing is sent to external servers. The Pulse does not require an internet connection to function.

---

## 4. Relationship to Existing Work

The Pulse is not the first system to attempt proactive AI behaviour. Existing approaches include cron-based scheduling (reliable but static), webhook/event-driven architectures (reactive by definition), and continuous LLM polling (effective but economically infeasible). The Pulse combines elements of anomaly detection (Layer 2's pattern recognition), event-driven architecture (Layer 1's event detection), and LLM agent design (Layer 3's question formation) in a novel hierarchical architecture specifically designed for the AI agent proactivity problem.

The closest related work is in IoT anomaly detection, where small LSTM and TCN models are routinely deployed on edge hardware to detect relevant patterns in sensor streams. The Pulse applies this established approach to the novel domain of AI agent perception.

---

## 5. Limitations and Future Work

**Rare events:** For events that occur very rarely (once a year), the cluster model will have sparse training data and low confidence. The module fingerprint prior provides a starting point, but rare-event prediction remains a known limitation.

**Adversarial signals:** A malicious file placed in a watched directory could trigger the Pulse. This is mitigated by the capability-based security model of the Macroa kernel (modules only watch directories they have declared), but it remains a potential attack surface worth further study.

**Transfer learning:** Future work could explore pre-training cluster models on aggregate anonymised data from multiple users (with consent) to improve cold-start performance, particularly for rare events.

---

## 6. Conclusion

The Pulse demonstrates that proactive AI agent behaviour does not require continuous LLM inference. By applying hierarchical perceptual processing — inspired by the brain's layered architecture — it is possible to achieve genuine proactivity at near-zero computational cost. The three-layer architecture (Retina, Limbic Filter, Prefrontal Filter) separates the cheap, fast work of change detection and pattern recognition from the expensive work of reasoning, ensuring that the LLM is only engaged when there is a specific, focused question to answer.

The Pulse is the foundation of the Macroa platform — an AI agent operating system designed around the principle that AI should be used only where no deterministic process can do the job. The Pulse embodies this principle: it achieves proactivity without AI, so that when AI is finally invoked, it is for the task it is uniquely suited to.

---

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
- Buckner, R. L., Andrews-Hanna, J. R., & Schacter, D. L. (2008). The Brain's Default Network. *Annals of the New York Academy of Sciences*, 1124(1), 1–38.
- Raichle, M. E. (2015). The Brain's Default Mode Network. *Annual Review of Neuroscience*, 38, 433–447.
