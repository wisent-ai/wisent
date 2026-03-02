"""GROM loss computation, direction constraints, and data collection."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.set import ContrastivePairSet
from wisent.core.constants import (
    NORM_EPS, GROM_LOSS_CONTRASTIVE_MARGIN, GROM_LOSS_CONTRASTIVE_WEIGHT,
    GROM_LOSS_UTILITY_WEIGHT, GROM_LOSS_CONCENTRATION_WEIGHT,
    GROM_LOSS_GATE_WARMUP_WEIGHT, GROM_LOSS_CAA_ALIGNMENT_WEIGHT,
)

def _compute_grom_loss_impl(
    self,
    direction_params: Dict[LayerName, nn.Parameter],
    effective_dirs: Dict[LayerName, torch.Tensor],
    pos_gate: torch.Tensor,
    neg_gate: torch.Tensor,
    pos_intensity: torch.Tensor,
    neg_intensity: torch.Tensor,
    data: Dict[str, Dict[LayerName, torch.Tensor]],
    layer_names: List[LayerName],
    step: int,
    direction_weight_params: Optional[Dict[LayerName, nn.Parameter]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the full GROM loss.

    Components:
    1. Contrastive loss: CRITICAL - maximize separation between pos and neg projections
    2. Behavior loss: Steering should be effective on positives
    3. Retain loss: Negatives should project LOWER than positives (not orthogonal!)
    4. Sparse loss: Encourage sparse layer activation
    5. Smooth loss: Penalize intensity variance across layers
    6. Independence loss: Directions should be independent
    7. Gate loss: Gate should discriminate pos from neg
    8. Direction utility loss: Differentiate directions based on effectiveness
    9. CAA alignment loss: Keep directions aligned with mean difference
    """
    loss_components = {}

    # 1. CONTRASTIVE SEPARATION LOSS - THE MOST IMPORTANT LOSS
    # This directly maximizes the gap: pos_proj - neg_proj
    contrastive_loss = torch.tensor(0.0)
    for layer in layer_names:
        pos_data = data["pos"][layer]
        neg_data = data["neg"][layer]
        eff_dir = effective_dirs[layer]

        # Compute projections (dot product with direction)
        pos_proj = (pos_data * eff_dir).sum(dim=1)  # [N_pos]
        neg_proj = (neg_data * eff_dir).sum(dim=1)  # [N_neg]

        # We want: pos_proj >> neg_proj (with margin)
        # Use pairwise margin loss: max(0, margin - (pos_proj - neg_proj))
        # Since we may have different numbers of pos/neg, use mean comparison
        pos_mean = pos_proj.mean()
        neg_mean = neg_proj.mean()

        # Margin should be proportional to the scale of activations
        # Typical projections are in range [-5, 5], so margin of 2.0 is reasonable
        margin = GROM_LOSS_CONTRASTIVE_MARGIN
        contrastive_loss = contrastive_loss + F.relu(margin - (pos_mean - neg_mean))

    contrastive_loss = contrastive_loss / len(layer_names)
    loss_components["contrastive"] = contrastive_loss

    # 2. Behavior loss - positives should have HIGH projection
    behavior_loss = torch.tensor(0.0)
    for layer in layer_names:
        pos_data = data["pos"][layer]
        eff_dir = effective_dirs[layer]
        pos_proj = (pos_data * eff_dir).sum(dim=1)
        # We want pos_proj > 0 (positive side of direction)
        behavior_loss = behavior_loss + F.relu(-pos_proj).mean()

    behavior_loss = behavior_loss / len(layer_names)
    loss_components["behavior"] = behavior_loss

    # 3. Retain loss - negatives should have LOWER projection than positives
    # Changed from abs() to direct negative projection encouragement
    retain_loss = torch.tensor(0.0)
    for layer in layer_names:
        neg_data = data["neg"][layer]
        eff_dir = effective_dirs[layer]
        neg_proj = (neg_data * eff_dir).sum(dim=1)
        # We want neg_proj < 0 (negative side of direction)
        retain_loss = retain_loss + F.relu(neg_proj).mean()

    retain_loss = retain_loss / len(layer_names)
    loss_components["retain"] = retain_loss

    # 4. Sparse loss - encourage sparse layer activation
    # Penalize uniform intensity distribution
    pos_intensity_norm = pos_intensity / (pos_intensity.sum(dim=1, keepdim=True) + NORM_EPS)
    sparse_loss = -torch.mean(torch.sum(pos_intensity_norm * torch.log(pos_intensity_norm + NORM_EPS), dim=1))
    sparse_loss = -sparse_loss  # We want LOW entropy (sparse)
    loss_components["sparse"] = sparse_loss

    # 5. Smooth loss - penalize abrupt intensity changes
    if pos_intensity.shape[1] > 1:
        intensity_diff = (pos_intensity[:, 1:] - pos_intensity[:, :-1]).abs()
        smooth_loss = intensity_diff.mean()
    else:
        smooth_loss = torch.tensor(0.0)
    loss_components["smooth"] = smooth_loss

    # 6. Independence loss - directions within manifold
    independence_loss = torch.tensor(0.0)
    for layer in layer_names:
        dirs = direction_params[layer]
        dirs_norm = F.normalize(dirs, p=2, dim=1)
        K = dirs_norm.shape[0]

        if K > 1:
            cos_sim = dirs_norm @ dirs_norm.T
            mask = 1 - torch.eye(K, device=cos_sim.device)

            # Penalize too high or too low similarity
            too_similar = F.relu(cos_sim - self.config.max_cosine_similarity)
            too_different = F.relu(self.config.min_cosine_similarity - cos_sim)
            independence_loss = independence_loss + ((too_similar + too_different) * mask).mean()

    independence_loss = independence_loss / len(layer_names)
    loss_components["independence"] = independence_loss

    # 6. Gate discrimination loss
    # Pos should have high gate (target=1), neg should have low gate (target=0)
    # Use BCE loss which provides gradient even when predictions are at 0.5
    # (The old relu-based loss had zero gradient at 0.5, causing the network to get stuck)
    gate_loss = (
        F.binary_cross_entropy(pos_gate, torch.ones_like(pos_gate)) +
        F.binary_cross_entropy(neg_gate, torch.zeros_like(neg_gate))
    )
    loss_components["gate"] = gate_loss

    # 7. Direction utility loss - reward directions that SEPARATE pos from neg
    # FIXED: Use (pos - neg) not (pos - abs(neg))
    direction_utility_loss = torch.tensor(0.0)
    if direction_weight_params is not None:
        for layer in layer_names:
            dirs = direction_params[layer]  # [K, H]
            dirs_norm = F.normalize(dirs, p=2, dim=1)
            K = dirs_norm.shape[0]

            if K > 1:
                pos_data = data["pos"][layer]  # [N, H]
                neg_data = data["neg"][layer]  # [N, H]

                # Compute per-direction projections
                pos_projs = pos_data @ dirs_norm.T  # [N, K]
                neg_projs = neg_data @ dirs_norm.T  # [N, K]

                # Per-direction utility: how well does this direction SEPARATE pos from neg?
                # FIXED: Use mean(pos) - mean(neg), NOT mean(pos) - mean(abs(neg))
                # Higher value = better separation (pos projects higher than neg)
                dir_utility = pos_projs.mean(dim=0) - neg_projs.mean(dim=0)  # [K]

                # Get current weights
                current_weights = F.softmax(direction_weight_params[layer], dim=0)

                # Weighted utility: reward putting weight on high-utility directions
                # Negate because we want to MAXIMIZE weighted utility (minimize negative)
                weighted_utility = -(current_weights * dir_utility).sum()
                direction_utility_loss = direction_utility_loss + weighted_utility

        direction_utility_loss = direction_utility_loss / len(layer_names)

    loss_components["direction_utility"] = direction_utility_loss

    # 8. Direction weight concentration loss - encourage non-uniform weights
    direction_concentration_loss = torch.tensor(0.0)
    if direction_weight_params is not None:
        for layer in layer_names:
            weights = F.softmax(direction_weight_params[layer], dim=0)
            K = weights.shape[0]
            if K > 1:
                # Negative entropy - encourages sparsity/concentration
                entropy = -(weights * torch.log(weights + NORM_EPS)).sum()
                max_entropy = torch.log(torch.tensor(float(K)))
                normalized_entropy = entropy / max_entropy

                # Also add concentration reward: maximize squared weights
                concentration = -(weights ** 2).sum()

                direction_concentration_loss = direction_concentration_loss + normalized_entropy + GROM_LOSS_CONCENTRATION_WEIGHT * concentration

        direction_concentration_loss = direction_concentration_loss / len(layer_names)

    loss_components["direction_concentration"] = direction_concentration_loss

    # 9. CAA alignment loss - keep primary direction aligned with mean difference
    # This ensures we don't drift away from the empirically-derived truthfulness direction
    caa_alignment_loss = torch.tensor(0.0)
    for layer in layer_names:
        pos_data = data["pos"][layer]
        neg_data = data["neg"][layer]
        dirs = direction_params[layer]

        # Compute CAA direction (mean difference)
        caa_dir = pos_data.mean(dim=0) - neg_data.mean(dim=0)
        caa_dir = F.normalize(caa_dir.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Primary direction (first direction, which was initialized with CAA)
        primary_dir = F.normalize(dirs[0:1], p=2, dim=1).squeeze(0)

        # Cosine similarity - we want it close to 1.0
        cos_sim = (primary_dir * caa_dir).sum()

        # Loss: penalize deviation from CAA direction
        caa_alignment_loss = caa_alignment_loss + (1.0 - cos_sim)

    caa_alignment_loss = caa_alignment_loss / len(layer_names)
    loss_components["caa_alignment"] = caa_alignment_loss

    # Combine losses with warmup
    # IMPORTANT: Contrastive loss is the PRIMARY loss - give it highest weight
    contrastive_weight = GROM_LOSS_CONTRASTIVE_WEIGHT
    utility_weight = GROM_LOSS_UTILITY_WEIGHT
    concentration_weight = GROM_LOSS_CONCENTRATION_WEIGHT
    caa_alignment_weight = GROM_LOSS_CAA_ALIGNMENT_WEIGHT

    if step < self.config.warmup_steps:
        # Warmup: focus on contrastive + CAA alignment
        total_loss = (
            contrastive_weight * contrastive_loss +
            self.config.behavior_weight * behavior_loss +
            self.config.retain_weight * retain_loss +
            caa_alignment_weight * caa_alignment_loss +
            GROM_LOSS_GATE_WARMUP_WEIGHT * gate_loss
        )
    else:
        # Full training with all losses
        total_loss = (
            contrastive_weight * contrastive_loss +
            self.config.behavior_weight * behavior_loss +
            self.config.retain_weight * retain_loss +
            self.config.sparse_weight * sparse_loss +
            self.config.smooth_weight * smooth_loss +
            self.config.independence_weight * independence_loss +
            gate_loss +
            utility_weight * direction_utility_loss +
            concentration_weight * direction_concentration_loss +
            caa_alignment_weight * caa_alignment_loss
        )

    return total_loss, loss_components

def _apply_direction_constraints_impl(self, directions: torch.Tensor) -> torch.Tensor:
    """Apply constraints to direction manifold."""
    # Normalize
    directions = F.normalize(directions, p=2, dim=1)

    # Cone constraint: all directions in same half-space as first
    if directions.shape[0] > 1:
        primary = directions[0:1]
        for i in range(1, directions.shape[0]):
            cos_sim = (directions[i:i+1] * primary).sum()
            if cos_sim < 0:
                directions[i] = -directions[i]

    return directions

def _collect_from_set_impl(
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

def get_training_logs_impl(self) -> List[Dict[str, Any]]:
    """Return training logs."""
    return self._training_logs
