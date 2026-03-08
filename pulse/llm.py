# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Built-in LLM handler — makes Pulse useful standalone by forwarding escalation
questions to an OpenAI-compatible model and printing the response.
"""

from __future__ import annotations

import os
import sys

import openai

from pulse.prefrontal import EscalationDecision


class SimpleLLMHandler:
    """
    Callable escalation handler that sends EscalationDecision.question to an
    OpenAI model and prints the result.

    Pass an instance directly to PulseRegistry.on_escalation():
        registry.on_escalation(SimpleLLMHandler())
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        self._model = model
        kwargs: dict = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def __call__(self, decision: EscalationDecision) -> None:
        if not decision.should_escalate:
            return

        question = decision.question
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": question}],
        )
        answer = response.choices[0].message.content

        print(f"\n--- Pulse escalation [{decision.module_id}] ---", file=sys.stdout)
        print(f"Q: {question}", file=sys.stdout)
        print(f"A: {answer}", file=sys.stdout)
        print("---", file=sys.stdout)
