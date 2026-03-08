# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Command-line entry point for standalone Pulse usage.

Usage:
    macroa-pulse --watch /some/dir --module fingerprint.json
    macroa-pulse --watch /some/dir --module fingerprint.json --provider anthropic
    macroa-pulse --watch /some/dir --module fingerprint.json --provider anthropic --model claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pulse.llm import SimpleLLMHandler
from pulse.registry import PulseRegistry

_PROVIDER_DEFAULTS = {
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-haiku-4-5",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="macroa-pulse",
        description="Run the Pulse proactive cognition subsystem standalone.",
    )
    parser.add_argument(
        "--watch",
        metavar="DIR",
        action="append",
        default=[],
        help="Directory to watch (may be repeated).",
    )
    parser.add_argument(
        "--module",
        metavar="FINGERPRINT_JSON",
        required=True,
        help="Path to a JSON file containing a module fingerprint.",
    )
    parser.add_argument(
        "--save-dir",
        metavar="PATH",
        default=str(Path.home() / ".macroa" / "pulse"),
        help="Directory in which to save model weights (default: ~/.macroa/pulse).",
    )
    parser.add_argument(
        "--provider",
        choices=list(_PROVIDER_DEFAULTS),
        default="openai",
        help="LLM provider to use for escalation responses (default: openai).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help="Override the default model for the chosen provider.",
    )
    args = parser.parse_args()

    fingerprint_path = Path(args.module)
    fingerprint_raw: dict = json.loads(fingerprint_path.read_text())
    module_id: str = fingerprint_raw.get("module_id", fingerprint_path.stem)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / f"{module_id}.pt"

    provider_cfg = _PROVIDER_DEFAULTS[args.provider]
    llm_model = args.model if args.model is not None else provider_cfg["model"]
    api_key = os.environ.get(provider_cfg["env_key"])
    handler = SimpleLLMHandler(
        api_key=api_key,
        model=llm_model,
        base_url=provider_cfg["base_url"],
    )

    registry = PulseRegistry(
        watch_dirs=args.watch,
        model_save_path=model_save_path,
    )
    registry.register_module(module_id, fingerprint_raw)
    registry.on_escalation(handler)
    registry.start()

    watch_display = ", ".join(args.watch) if args.watch else "(none)"
    print(
        f"Pulse running | watching: {watch_display} | "
        f"provider: {args.provider} | model: {llm_model}",
        file=sys.stdout,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        registry.stop()
        print("Pulse stopped.", file=sys.stdout)
