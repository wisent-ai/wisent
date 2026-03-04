"""TETNO layer scaling and threshold learning."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.control.steering_methods.methods.advanced._tetno_types import TETNOConfig, TETNOResult

class TETNOScalingMixin:
    """Mixin: layer scaling and threshold learning."""

    def _compute_layer_scales(
        self,
        behavior_vectors: Dict[LayerName, torch.Tensor],
        pair_set: ContrastivePairSet,
    ) -> Dict[LayerName, float]:
        """
        Compute per-layer scaling factors.
        
        Uses the separation score at each layer to determine relative importance.
        """
        if not self.config.per_layer_scaling:
            return {layer: 1.0 for layer in behavior_vectors.keys()}
        
        buckets = self._collect_from_set(pair_set)
        layer_scores: Dict[LayerName, float] = {}
        
        for layer_name, vec in behavior_vectors.items():
            if layer_name not in buckets:
                layer_scores[layer_name] = 1.0
                continue
            
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                layer_scores[layer_name] = 1.0
                continue
            
            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            
            # Compute separation score
            vec_norm = F.normalize(vec, p=2, dim=0)
            pos_proj = (pos_tensor * vec_norm).sum(dim=1).mean()
            neg_proj = (neg_tensor * vec_norm).sum(dim=1).mean()
            separation = (pos_proj - neg_proj).item()
            
            layer_scores[layer_name] = max(self.config.min_layer_scale, separation)
        
        # Normalize scales to have mean 1.0
        if layer_scores:
            mean_score = sum(layer_scores.values()) / len(layer_scores)
            if mean_score > 0:
                layer_scores = {k: v / mean_score for k, v in layer_scores.items()}
        
        return layer_scores
    
    def _learn_threshold(
        self,
        condition_vector: torch.Tensor,
        pair_set: ContrastivePairSet,
    ) -> float:
        """
        Learn optimal threshold via grid search.
        
        Finds threshold that best separates pos (should steer) from neg (shouldn't).
        """
        buckets = self._collect_from_set(pair_set)
        
        # Find sensor layer activations
        sensor_layer_name = None
        for layer_name in buckets.keys():
            try:
                layer_idx = int(layer_name.split("_")[-1]) if "_" in str(layer_name) else int(layer_name)
                if layer_idx == self.config.sensor_layer:
                    sensor_layer_name = layer_name
                    break
            except (ValueError, AttributeError):
                continue
        
        if sensor_layer_name is None:
            return self.config.condition_threshold
        
        pos_list, neg_list = buckets[sensor_layer_name]
        if not pos_list or not neg_list:
            return self.config.condition_threshold
        
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Compute similarities
        c_norm = F.normalize(condition_vector, p=2, dim=0)
        pos_norm = F.normalize(pos_tensor, p=2, dim=1)
        neg_norm = F.normalize(neg_tensor, p=2, dim=1)
        
        pos_sims = (pos_norm * c_norm).sum(dim=1)
        neg_sims = (neg_norm * c_norm).sum(dim=1)
        
        # Grid search for best threshold
        min_sim = min(pos_sims.min().item(), neg_sims.min().item())
        max_sim = max(pos_sims.max().item(), neg_sims.max().item())
        
        best_threshold = self.config.condition_threshold
        best_accuracy = 0.0
        
        for i in range(self.config.threshold_search_steps):
            threshold = min_sim + (max_sim - min_sim) * i / self.config.threshold_search_steps
            
            # Accuracy: pos should be above threshold, neg below
            pos_correct = (pos_sims > threshold).float().mean().item()
            neg_correct = (neg_sims <= threshold).float().mean().item()
            accuracy = (pos_correct + neg_correct) / 2
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self._training_logs.append({
            "phase": "threshold_search",
            "best_threshold": best_threshold,
            "best_accuracy": best_accuracy,
            "pos_sim_range": [pos_sims.min().item(), pos_sims.max().item()],
            "neg_sim_range": [neg_sims.min().item(), neg_sims.max().item()],
        })
        
        return best_threshold
    
    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Build {layer_name: ([pos tensors...], [neg tensors...])} from pairs."""
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
    
    def get_training_logs(self) -> List[Dict[str, Any]]:
        """Return training logs."""
        return self._training_logs


