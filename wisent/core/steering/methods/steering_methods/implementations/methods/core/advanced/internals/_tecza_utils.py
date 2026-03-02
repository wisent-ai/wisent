"""TECZA utility methods."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

class TECZAUtilsMixin:
    """Mixin: constraints, metadata, and data collection."""

    def _apply_constraints(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Apply constraints to directions after gradient step.
        """
        # Normalize
        directions = F.normalize(directions, p=2, dim=1)
        
        # Cone constraint: ensure all directions have positive correlation with first
        if self.config.cone_constraint and directions.shape[0] > 1:
            primary = directions[0:1]  # [1, H]
            for i in range(1, directions.shape[0]):
                cos_sim = (directions[i:i+1] * primary).sum()
                if cos_sim < 0:
                    # Flip direction to be in same half-space
                    directions[i] = -directions[i]
        
        return directions
    
    def _compute_direction_metadata(
        self,
        directions: torch.Tensor,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute metadata about the discovered directions.
        """
        K = directions.shape[0]
        dirs_normalized = F.normalize(directions, p=2, dim=1)
        
        # Pairwise cosine similarities
        cos_sim_matrix = dirs_normalized @ dirs_normalized.T
        
        # Per-direction separation scores
        pos_proj = pos_tensor @ dirs_normalized.T
        neg_proj = neg_tensor @ dirs_normalized.T
        separation_scores = (pos_proj.mean(dim=0) - neg_proj.mean(dim=0)).tolist()
        
        # Average cosine similarity (off-diagonal)
        if K > 1:
            mask = 1 - torch.eye(K)
            avg_cos_sim = (cos_sim_matrix * mask).sum() / (K * (K - 1))
        else:
            avg_cos_sim = 1.0
        
        return {
            "num_directions": K,
            "separation_scores": separation_scores,
            "avg_cosine_similarity": float(avg_cos_sim),
            "cosine_similarity_matrix": cos_sim_matrix.tolist(),
            "direction_norms": directions.norm(dim=1).tolist(),
        }
    
    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Build {layer_name: ([pos tensors...], [neg tensors...])} by iterating pairs.
        """
        from collections import defaultdict
        
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]] = defaultdict(lambda: ([], []))
        
        for pair in pair_set.pairs:
            pos_la = getattr(pair.positive_response, "layers_activations", None)
            neg_la = getattr(pair.negative_response, "layers_activations", None)
            
            if pos_la is None or neg_la is None:
                continue
            
            layer_names = set(pos_la.to_dict().keys()) | set(neg_la.to_dict().keys())
            for layer in layer_names:
                p = pos_la.to_dict().get(layer, None) if pos_la is not None else None
                n = neg_la.to_dict().get(layer, None) if neg_la is not None else None
                if isinstance(p, torch.Tensor) and isinstance(n, torch.Tensor):
                    buckets[layer][0].append(p)
                    buckets[layer][1].append(n)
        
        return buckets
    
    def get_training_logs(self) -> List[Dict[str, float]]:
        """Return the training logs from the last train() call."""
        return self._training_logs
