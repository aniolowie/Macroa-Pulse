# Macroa — Claude Code Context

## What This Project Is
Macroa is an AI agent operating system. It is not a chatbot framework or an agent library.
It is a platform that other agents run on, the way apps run on Windows.

## Design Philosophy
AI is used only where no deterministic process can do the job. Everything else is infrastructure.

## Current Build Phase
[UPDATE THIS EVERY SESSION]
Phase: Pulse — implementing pulse/retina.py

## Architecture Documents
- 1_Pulse_Architecture_Guide.md — Pulse specification (build this first)
- 3_Post_Pulse_Architecture.md — Everything after the Pulse

## Naming Conventions
- Modules: things that run on Macroa (not "agents", not "plugins")
- MICRO / EDGE / CORE / APEX: LLM compute tiers (not "nano/haiku/sonnet/opus")
- Pulse: the proactive cognition subsystem
- Kernel: the always-on background daemon
- Shell: the Textual TUI (the face of the OS)
- BIOS: the one-time setup

## What Claude Should Never Do
- Make architectural decisions not in the spec documents
- Merge files that the spec lists as separate
- Add dependencies not listed in the spec
- Change security model (capability-based, manifest-declared)
- Refactor across module boundaries
