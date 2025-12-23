from __future__ import annotations

from typing import List
import torch
import numpy as np

from wisent.core.steering_methods.core.atoms import PerLayerBaseSteeringMethod
from wisent.core.errors import InsufficientDataError
from wisent.core.utils.device import preferred_dtype

__all__ = [
    "HyperplaneMethod",
]


class HyperplaneMethod(PerLayerBaseSteeringMethod):
    """
    Hyperplane-based steering using classifier decision boundary.
    
    Instead of computing mean(pos) - mean(neg) like CAA, this method trains
    a logistic regression classifier to separate positive from negative activations,
    then uses the classifier's weight vector (hyperplane normal) as the steering vector.
    
    This works better when the geometry is orthogonal (each contrastive pair has
    a unique direction) rather than linear (all pairs share a common direction).
    In orthogonal geometry, CAA's mean difference cancels out to near-zero,
    while the classifier can still find a separating hyperplane.
    """
    name = "hyperplane"
    description = "Classifier-based steering using logistic regression decision boundary as steering vector."

    def train_for_layer(self, pos_list: List[torch.Tensor], neg_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Train hyperplane steering vector for a single layer using logistic regression.

        arguments:
            pos_list: List of positive activations (torch.Tensor) for this layer.
            neg_list: List of negative activations (torch.Tensor) for this layer.
            
        returns:
            torch.Tensor steering vector for the layer (classifier weights / hyperplane normal).
        """
        if not pos_list or not neg_list:
            raise InsufficientDataError(reason="Both positive and negative lists must be non-empty.")
        
        pos = torch.stack([t.detach().to("cpu").float().reshape(-1) for t in pos_list], dim=0)
        neg = torch.stack([t.detach().to("cpu").float().reshape(-1) for t in neg_list], dim=0)
        
        pos_np = pos.numpy()
        neg_np = neg.numpy()
        
        X = np.vstack([pos_np, neg_np])
        y = np.array([1] * len(pos_np) + [0] * len(neg_np))
        
        # Train logistic regression classifier
        from sklearn.linear_model import LogisticRegression
        
        max_iter = int(self.kwargs.get("max_iter", 1000))
        C = float(self.kwargs.get("C", 1.0))
        
        clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
        clf.fit(X, y)
        
        # Use classifier weights as steering vector
        v = torch.tensor(clf.coef_[0], dtype=preferred_dtype())
        
        if bool(self.kwargs.get("normalize", True)):
            v = self._safe_l2_normalize(v)
        
        return v

    def _safe_l2_normalize(self, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if v.ndim != 1:
            v = v.reshape(-1)
        return v / (torch.linalg.norm(v) + eps)
