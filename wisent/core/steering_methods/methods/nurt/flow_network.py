"""
Flow velocity network and Euler integration for Concept Flow steering.

The velocity network learns a velocity field v(z_t, t) that transports
activations from the negative distribution to the positive distribution
in the low-dimensional concept subspace discovered by SVD.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


__all__ = [
    "FlowVelocityNetwork",
    "euler_integrate",
]


class FlowVelocityNetwork(nn.Module):
    """
    Velocity network for conditional flow matching in concept subspace.

    Input: concatenation of z_t (concept_dim) and t (1) -> velocity (concept_dim).
    Architecture is intentionally tiny since concept_dim is typically 3-10.

    Args:
        concept_dim: Dimensionality of the concept subspace (k).
        hidden_dim: Hidden layer dimension. If None, auto-computed as
                     max(32, min(128, 4 * concept_dim)).
    """

    def __init__(self, concept_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.concept_dim = concept_dim
        if hidden_dim is None:
            hidden_dim = max(32, min(128, 4 * concept_dim))
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(concept_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, concept_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at (z_t, t).

        Args:
            z_t: Current position in concept subspace [batch, concept_dim] or [concept_dim].
            t: Time scalar [batch, 1] or [1] or scalar tensor.

        Returns:
            Predicted velocity [batch, concept_dim].
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)

        # Ensure t is [batch, 1]
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).expand(z_t.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
            if t.shape[0] == 1 and z_t.shape[0] > 1:
                t = t.expand(z_t.shape[0], 1)

        inp = torch.cat([z_t, t], dim=-1)
        return self.net(inp)


def euler_integrate(
    network: FlowVelocityNetwork,
    z_0: torch.Tensor,
    t_max: float = 1.0,
    num_steps: int = 4,
) -> torch.Tensor:
    """
    Euler integration of the learned velocity field.

    Integrates from t=0 to t=t_max in num_steps steps:
        z_{i+1} = z_i + dt * v(z_i, t_i)

    Args:
        network: Trained FlowVelocityNetwork.
        z_0: Starting point in concept subspace [batch, k] or [k].
        t_max: Integration endpoint (controls steering strength).
        num_steps: Number of Euler steps.

    Returns:
        z_final after integration [batch, k] or [k].
    """
    squeeze = z_0.dim() == 1
    if squeeze:
        z_0 = z_0.unsqueeze(0)

    dt = t_max / num_steps
    z = z_0.clone()

    for i in range(num_steps):
        t_val = i * dt
        t_tensor = torch.full(
            (z.shape[0], 1), t_val, device=z.device, dtype=z.dtype
        )
        with torch.no_grad():
            v = network(z, t_tensor)
        z = z + dt * v

    if squeeze:
        z = z.squeeze(0)
    return z
