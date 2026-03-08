# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Maciej (aniolowie) , https://github.com/aniolowie/Macroa-Pulse

"""
Layer 2: Limbic , per-module relevance scoring via small LSTM models.
Architecture: docs/ARCHITECTURE.md §Layer 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pulse.fingerprint import ModuleFingerprint
from pulse.retina import FEATURE_DIM, SignalEvent


class ClusterModel(nn.Module):
    """
    Small per-module LSTM that scores a window of SignalEvent feature vectors
    for relevance to a specific module.

    Input:  (batch=1, window_len, FEATURE_DIM) float32
    Output: scalar float32 in [0.0, 1.0]
    """

    HIDDEN_SIZE: int = 64

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(self.HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, window_len, FEATURE_DIM) float32
        Returns:
            scalar tensor, float32 relevance score in [0.0, 1.0]
        """
        _, (h_n, _) = self.lstm(x)           # h_n: (1, 1, HIDDEN_SIZE)
        last_hidden = h_n.squeeze(0).squeeze(0)  # (HIDDEN_SIZE,)
        score = self.sigmoid(self.head(last_hidden))
        return score.squeeze()


@dataclass
class _Entry:
    model: ClusterModel
    optimizer: torch.optim.Adam


class LimbicLayer:
    """
    Registry of per-module ClusterModel instances.

    Each module gets its own independent LSTM that learns, online, whether a
    window of SignalEvents is relevant enough to wake the module.
    """

    def __init__(self) -> None:
        self._registry: dict[str, _Entry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, module_id: str, fingerprint: ModuleFingerprint) -> None:
        """
        Create a ClusterModel for the module and apply cold-start weight biasing
        derived from fingerprint.slot_relevance_mask().

        Relevant slots have their LSTM input weights scaled up; irrelevant slots
        have them scaled down so the model starts with a meaningful prior.
        """
        model = ClusterModel()
        self._apply_cold_start_bias(model, fingerprint.slot_relevance_mask())
        with torch.no_grad():
            t = fingerprint.default_threshold
            model.head.bias.data = torch.tensor(
                [math.log(t / (1 - t + 1e-9))]
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self._registry[module_id] = _Entry(model=model, optimizer=optimizer)

    def score(self, module_id: str, window: list[SignalEvent]) -> float:
        """
        Run inference on a window of SignalEvents.

        Returns 0.0 if the window is empty or the module is not registered.
        """
        if not window or module_id not in self._registry:
            return 0.0
        entry = self._registry[module_id]
        entry.model.eval()
        x = self._window_to_tensor(window)
        with torch.no_grad():
            result = entry.model(x)
        return float(result.item())

    def update_weights(
        self,
        module_id: str,
        window: list[SignalEvent],
        label: float,
    ) -> None:
        """
        Perform a single online gradient step using BCELoss.

        No-op if the window is empty or the module is not registered.
        """
        if not window or module_id not in self._registry:
            return
        entry = self._registry[module_id]
        entry.model.train()
        x = self._window_to_tensor(window)
        target = torch.tensor(label, dtype=torch.float32)
        prediction = entry.model(x)
        loss = nn.functional.binary_cross_entropy(prediction, target)
        entry.optimizer.zero_grad()
        loss.backward()
        entry.optimizer.step()

    def save(self, path: Path) -> None:
        """Persist all model weights and optimiser states to disk."""
        checkpoint = {
            module_id: {
                "model": entry.model.state_dict(),
                "optimizer": entry.optimizer.state_dict(),
            }
            for module_id, entry in self._registry.items()
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        """
        Restore model weights and optimiser states from disk.

        Modules present in the checkpoint but not yet registered are
        re-created as fresh ClusterModel instances with restored state.
        """
        checkpoint = torch.load(path, weights_only=True)
        for module_id, states in checkpoint.items():
            if module_id not in self._registry:
                model = ClusterModel()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                self._registry[module_id] = _Entry(model=model, optimizer=optimizer)
            entry = self._registry[module_id]
            entry.model.load_state_dict(states["model"])
            entry.optimizer.load_state_dict(states["optimizer"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cold_start_bias(model: ClusterModel, mask: np.ndarray) -> None:
        """
        Scale the LSTM input-to-hidden weight columns by a factor derived from
        the slot relevance mask.

        Scale formula: 0.1 + 1.9 * mask[i]
            mask = 0.0  ->  scale = 0.1  (nearly zeroed, irrelevant slot)
            mask = 0.5  ->  scale = 1.05 (neutral)
            mask = 1.0  ->  scale = 2.0  (doubled, highly relevant slot)

        weight_ih_l0 shape: (4 * HIDDEN_SIZE, FEATURE_DIM)
        Each column corresponds to one input feature slot.
        """
        scale = torch.tensor(0.1 + 1.9 * mask, dtype=torch.float32)  # (FEATURE_DIM,)
        with torch.no_grad():
            # weight_ih_l0: (4*H, FEATURE_DIM) — broadcast-multiply each column
            model.lstm.weight_ih_l0.mul_(scale.unsqueeze(0))

    @staticmethod
    def _window_to_tensor(window: list[SignalEvent]) -> torch.Tensor:
        """Convert a list of SignalEvents to a (1, T, FEATURE_DIM) float32 tensor."""
        vectors = np.stack([e.to_feature_vector() for e in window], axis=0)
        return torch.from_numpy(vectors).unsqueeze(0)  # (1, T, FEATURE_DIM)
