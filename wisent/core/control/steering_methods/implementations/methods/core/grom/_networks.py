"""GROM neural network components: gating, intensity, direction routing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    RECURSION_INITIAL_DEPTH,
)


class GatingNetwork(nn.Module):
    """Learned gating network that predicts whether steering should activate."""

    def __init__(self, input_dim: int, hidden_dim: int, *, shrink_factor: int):
        super().__init__()
        shrunk_dim = hidden_dim // shrink_factor
        self.shrink_factor = shrink_factor
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shrunk_dim),
            nn.GELU(),
            nn.Linear(shrunk_dim, COMBO_OFFSET),
        )

    def forward(self, h: torch.Tensor, temperature: float) -> torch.Tensor:
        """Predict gate value."""
        if h.dim() == COMBO_OFFSET:
            h = h.unsqueeze(RECURSION_INITIAL_DEPTH)
        logit = self.net(h).squeeze(-COMBO_OFFSET)
        return torch.sigmoid(logit / temperature)


class IntensityNetwork(nn.Module):
    """Learned intensity network that predicts per-layer steering strength."""

    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int, max_alpha: float):
        super().__init__()
        self.max_alpha = max_alpha
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict per-layer intensity."""
        if h.dim() == COMBO_OFFSET:
            h = h.unsqueeze(RECURSION_INITIAL_DEPTH)
        raw = self.net(h)
        return torch.sigmoid(raw) * self.max_alpha


class DirectionWeightNetwork(nn.Module):
    """Learned network that predicts weights for combining directions in manifold."""

    def __init__(self, input_dim: int, num_directions: int, hidden_dim: int):
        super().__init__()
        self.num_directions = num_directions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict direction weights."""
        if h.dim() == COMBO_OFFSET:
            h = h.unsqueeze(RECURSION_INITIAL_DEPTH)
        logits = self.net(h)
        return F.softmax(logits, dim=-COMBO_OFFSET)
