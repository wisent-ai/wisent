"""Advanced steering methods beyond linear activation addition."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SteeringResult:
    """Result from applying a steering method."""
    method_name: str
    base_truthful: int
    steered_truthful: int
    total: int
    improvement: int
    details: Dict


class SteeringMethod(ABC):
    """Base class for steering methods."""

    @abstractmethod
    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        """Fit the steering method on reference data."""
        pass

    @abstractmethod
    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        """Transform activations using this steering method."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LinearSteering(SteeringMethod):
    """Standard linear steering: activation + strength * direction."""

    def __init__(self, strength: float = 1.0):
        self.strength = strength
        self.direction = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        self.direction = torch.from_numpy(direction).float()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        return activations + self.strength * self.direction.to(activations.device)

    @property
    def name(self) -> str:
        return f"linear_s{self.strength}"


class ClampingSteering(SteeringMethod):
    """Clamp activations to stay within truthful region bounds."""

    def __init__(self, margin: float = 0.1):
        self.margin = margin
        self.pos_min = None
        self.pos_max = None
        self.direction = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        self.pos_min = torch.from_numpy(pos_acts.min(axis=0) - self.margin * np.abs(pos_acts.min(axis=0))).float()
        self.pos_max = torch.from_numpy(pos_acts.max(axis=0) + self.margin * np.abs(pos_acts.max(axis=0))).float()
        direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        self.direction = torch.from_numpy(direction / (np.linalg.norm(direction) + 1e-8)).float()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        clamped = torch.clamp(activations, self.pos_min.to(device), self.pos_max.to(device))
        return clamped

    @property
    def name(self) -> str:
        return "clamping"


class ProjectionSteering(SteeringMethod):
    """Project activations away from untruthful subspace."""

    def __init__(self, n_components: int = 5, strength: float = 1.0):
        self.n_components = n_components
        self.strength = strength
        self.untruthful_basis = None
        self.truthful_centroid = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        from sklearn.decomposition import PCA
        # Find principal components of untruthful activations
        pca = PCA(n_components=min(self.n_components, len(neg_acts) - 1))
        pca.fit(neg_acts)
        self.untruthful_basis = torch.from_numpy(pca.components_).float()
        self.truthful_centroid = torch.from_numpy(pos_acts.mean(axis=0)).float()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        basis = self.untruthful_basis.to(device)
        centroid = self.truthful_centroid.to(device)
        # Project out untruthful components
        for component in basis:
            proj = torch.sum(activations * component, dim=-1, keepdim=True) * component
            activations = activations - self.strength * proj
        # Shift toward truthful centroid
        activations = activations + self.strength * 0.1 * (centroid - activations)
        return activations

    @property
    def name(self) -> str:
        return f"projection_n{self.n_components}"


class ReplacementSteering(SteeringMethod):
    """Replace activations with interpolation toward truthful prototype."""

    def __init__(self, blend: float = 0.5):
        self.blend = blend
        self.truthful_prototype = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        self.truthful_prototype = torch.from_numpy(pos_acts.mean(axis=0)).float()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        prototype = self.truthful_prototype.to(device)
        return (1 - self.blend) * activations + self.blend * prototype

    @property
    def name(self) -> str:
        return f"replacement_b{self.blend}"


class ContrastSteering(SteeringMethod):
    """Maximize distance from untruthful centroid while preserving norm."""

    def __init__(self, strength: float = 1.0):
        self.strength = strength
        self.neg_centroid = None
        self.pos_centroid = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        self.neg_centroid = torch.from_numpy(neg_acts.mean(axis=0)).float()
        self.pos_centroid = torch.from_numpy(pos_acts.mean(axis=0)).float()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        neg_c = self.neg_centroid.to(device)
        pos_c = self.pos_centroid.to(device)
        # Direction away from negative centroid
        away_dir = activations - neg_c
        away_dir = away_dir / (torch.norm(away_dir, dim=-1, keepdim=True) + 1e-8)
        # Also direction toward positive
        toward_dir = pos_c - activations
        toward_dir = toward_dir / (torch.norm(toward_dir, dim=-1, keepdim=True) + 1e-8)
        # Combined push
        orig_norm = torch.norm(activations, dim=-1, keepdim=True)
        steered = activations + self.strength * (away_dir + toward_dir) * orig_norm * 0.1
        return steered

    @property
    def name(self) -> str:
        return f"contrast_s{self.strength}"


class MLPSteering(SteeringMethod):
    """Learned non-linear steering via small MLP."""

    def __init__(self, hidden_dim: int = 256, epochs: int = 100):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.mlp = None
        self.input_dim = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        self.input_dim = pos_acts.shape[1]
        # MLP that transforms activations toward truthful distribution
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        # Train to map neg -> pos direction
        X = torch.from_numpy(neg_acts).float()
        # Target: shift toward pos centroid
        pos_centroid = torch.from_numpy(pos_acts.mean(axis=0)).float()
        Y = pos_centroid.unsqueeze(0).expand(len(neg_acts), -1)

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.mlp(X)
            loss = nn.MSELoss()(pred, Y)
            loss.backward()
            optimizer.step()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        dtype = activations.dtype
        self.mlp = self.mlp.to(device).to(dtype)
        with torch.no_grad():
            transformed = self.mlp(activations)
        return transformed

    @property
    def name(self) -> str:
        return f"mlp_h{self.hidden_dim}"


class AdaptiveSteering(SteeringMethod):
    """Adaptive steering that adjusts strength based on distance from boundary."""

    def __init__(self, max_strength: float = 2.0):
        self.max_strength = max_strength
        self.classifier = None
        self.direction = None

    def fit(self, pos_acts: np.ndarray, neg_acts: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        X = np.vstack([pos_acts, neg_acts])
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
        self.classifier = LogisticRegression()
        self.classifier.fit(X, y)
        direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        self.direction = torch.from_numpy(direction / (np.linalg.norm(direction) + 1e-8)).float()

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        acts_np = activations.cpu().numpy()
        # Get probability of being truthful
        probs = self.classifier.predict_proba(acts_np)[:, 1]
        # Adaptive strength: steer more if far from truthful
        strengths = self.max_strength * (1 - probs)
        strengths = torch.from_numpy(strengths).float().to(device).unsqueeze(-1)
        return activations + strengths * self.direction.to(device)

    @property
    def name(self) -> str:
        return f"adaptive_s{self.max_strength}"


def get_all_steering_methods(strengths: List[float] = None) -> List[SteeringMethod]:
    """Get all available steering methods with various configurations."""
    if strengths is None:
        strengths = [0.5, 1.0, 2.0, 5.0]

    methods = []
    for s in strengths:
        methods.append(LinearSteering(strength=s))
    methods.append(ClampingSteering(margin=0.1))
    methods.append(ClampingSteering(margin=0.5))
    for n in [3, 5, 10]:
        methods.append(ProjectionSteering(n_components=n, strength=1.0))
    for b in [0.3, 0.5, 0.7, 0.9]:
        methods.append(ReplacementSteering(blend=b))
    for s in strengths:
        methods.append(ContrastSteering(strength=s))
    methods.append(MLPSteering(hidden_dim=128, epochs=50))
    methods.append(MLPSteering(hidden_dim=256, epochs=100))
    for s in [1.0, 2.0, 5.0]:
        methods.append(AdaptiveSteering(max_strength=s))

    return methods
