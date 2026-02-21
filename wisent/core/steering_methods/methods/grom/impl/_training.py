"""GROM training helpers: geometry analysis, direction init, data prep."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    def _analyze_and_adapt_geometry(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
        hidden_dim: int,
    ) -> GeometryAdaptation:
        """
        Analyze geometry of activations and adapt GROM configuration.
        
        Returns:
            GeometryAdaptation with detected structure and adaptations made.
        """
        from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
            detect_geometry_structure,
            GeometryAnalysisConfig,
        )
        
        # Find the layer to analyze
        analysis_layer_idx = self.config.geometry_analysis_layer or self.config.sensor_layer
        analysis_layer = None
        for layer in layer_names:
            try:
                idx = int(str(layer).split("_")[-1])
                if idx == analysis_layer_idx:
                    analysis_layer = layer
                    break
            except (ValueError, IndexError):
                continue
        
        if analysis_layer is None:
            analysis_layer = layer_names[len(layer_names) // 2]
        
        # Get activations for analysis
        pos_list, neg_list = buckets[analysis_layer]
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Run geometry detection
        geo_config = GeometryAnalysisConfig(
            num_components=self.config.num_directions,
            max_clusters=5,
            manifold_neighbors=min(10, len(pos_list) - 1),
        )
        geo_result = detect_geometry_structure(pos_tensor, neg_tensor, geo_config)
        
        # Extract scores
        structure_scores = {
            name: score.score for name, score in geo_result.all_scores.items()
        }
        detected_structure = geo_result.best_structure.value
        
        # Determine adaptations
        adaptations = []
        original_num_directions = self.config.num_directions
        adapted_num_directions = original_num_directions
        gating_enabled = True
        
        linear_score = structure_scores.get("linear", 0)
        cone_score = structure_scores.get("cone", 0)
        manifold_score = structure_scores.get("manifold", 0)
        
        # Adaptation 1: Simplify if linear
        if linear_score > self.config.linear_threshold:
            if self.config.auto_num_directions:
                adapted_num_directions = 1
                adaptations.append(f"Reduced num_directions to 1 (linear score={linear_score:.2f})")
            
            if self.config.skip_gating_if_linear:
                gating_enabled = False
                adaptations.append("Disabled gating network (linear structure)")
        
        # Adaptation 2: Adjust directions based on cone structure
        elif cone_score > 0.7 and self.config.auto_num_directions:
            # Cone structure benefits from multiple directions
            cone_details = geo_result.all_scores.get("cone")
            if cone_details and hasattr(cone_details, "details"):
                sig_dirs = cone_details.details.get("significant_directions", 3)
                adapted_num_directions = max(2, min(sig_dirs + 1, 7))
                if adapted_num_directions != original_num_directions:
                    adaptations.append(
                        f"Adjusted num_directions to {adapted_num_directions} based on cone structure"
                    )
        
        # Adaptation 3: Increase directions for manifold/orthogonal
        elif (manifold_score > 0.8 or structure_scores.get("orthogonal", 0) > 0.7):
            if self.config.auto_num_directions and adapted_num_directions < 5:
                adapted_num_directions = 5
                adaptations.append(
                    f"Increased num_directions to 5 for manifold/orthogonal structure"
                )
        
        if not adaptations:
            adaptations.append("No adaptations needed - using default configuration")
        
        return GeometryAdaptation(
            detected_structure=detected_structure,
            structure_scores=structure_scores,
            adaptations_made=adaptations,
            original_num_directions=original_num_directions,
            adapted_num_directions=adapted_num_directions,
            gating_enabled=gating_enabled,
            recommendation=geo_result.recommendation,
        )
    
    def _initialize_directions(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
        hidden_dim: int,
        num_directions: Optional[int] = None,
    ) -> Dict[LayerName, torch.Tensor]:
        """Initialize direction manifold for each layer using PCA for diversity."""
        directions = {}
        K = num_directions if num_directions is not None else self.config.num_directions

        for layer in layer_names:
            pos_list, neg_list = buckets[layer]

            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

            # First direction: CAA (mean difference)
            caa_dir = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
            caa_dir = F.normalize(caa_dir, p=2, dim=0)

            # Use PCA on difference vectors for diverse directions
            diff_vectors = pos_tensor - neg_tensor  # [N, H]

            # Center the difference vectors
            diff_centered = diff_vectors - diff_vectors.mean(dim=0, keepdim=True)

            # SVD for PCA
            try:
                # Use svd_lowrank for efficiency on large hidden_dim
                U, S, V = torch.svd_lowrank(diff_centered, q=min(K + 2, diff_centered.shape[0], diff_centered.shape[1]))
                # V contains principal components: [H, K+2]
                pca_dirs = V.T  # [K+2, H]
            except Exception:
                # Fallback to random orthogonal initialization
                pca_dirs = None

            dirs = torch.zeros(K, hidden_dim)
            dirs[0] = caa_dir  # First direction is always CAA

            if pca_dirs is not None and K > 1:
                # Use PCA directions for remaining directions
                # Skip the first PCA component if it's too similar to CAA
                pca_idx = 0
                for i in range(1, K):
                    if pca_idx < pca_dirs.shape[0]:
                        candidate = F.normalize(pca_dirs[pca_idx], p=2, dim=0)
                        cos_with_caa = (candidate * caa_dir).sum().abs()

                        # If too similar to CAA, try next PCA component
                        if cos_with_caa > 0.9 and pca_idx + 1 < pca_dirs.shape[0]:
                            pca_idx += 1
                            candidate = F.normalize(pca_dirs[pca_idx], p=2, dim=0)

                        dirs[i] = candidate
                        pca_idx += 1
                    else:
                        # Fallback: orthogonalize random vector
                        random_dir = torch.randn(hidden_dim)
                        # Gram-Schmidt orthogonalization against previous directions
                        for j in range(i):
                            proj = (random_dir * dirs[j]).sum() * dirs[j]
                            random_dir = random_dir - proj
                        dirs[i] = F.normalize(random_dir, p=2, dim=0)
            else:
                # No PCA available, use Gram-Schmidt orthogonalization
                for i in range(1, K):
                    random_dir = torch.randn(hidden_dim)
                    for j in range(i):
                        proj = (random_dir * dirs[j]).sum() * dirs[j]
                        random_dir = random_dir - proj
                    dirs[i] = F.normalize(random_dir, p=2, dim=0)

            # Ensure all in same half-space as CAA
            for i in range(1, K):
                if (dirs[i] * caa_dir).sum() < 0:
                    dirs[i] = -dirs[i]

            directions[layer] = dirs

        return directions
    
    def _prepare_data_tensors(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
    ) -> Dict[str, Dict[LayerName, torch.Tensor]]:
        """Prepare stacked tensors for training."""
        data = {"pos": {}, "neg": {}}
        
        for layer in layer_names:
            pos_list, neg_list = buckets[layer]
            data["pos"][layer] = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            data["neg"][layer] = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        return data
    
    def _find_sensor_layer(self, layer_names: List[LayerName]) -> LayerName:
        """Find the sensor layer from available layers."""
        for layer in layer_names:
            try:
                layer_idx = int(str(layer).split("_")[-1])
                if layer_idx == self.config.sensor_layer:
                    return layer
            except (ValueError, IndexError):
                continue
        
        # Fallback to middle layer
        return layer_names[len(layer_names) // 2]
    
