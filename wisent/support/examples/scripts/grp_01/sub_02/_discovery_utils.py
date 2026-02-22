"""
Utility functions and dataclasses for direction discovery.

Contains S3 helpers, data loading, and result dataclasses.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


S3_BUCKET = "wisent-bucket"
S3_PREFIX = "direction_discovery"


def s3_sync_download(model_name: str, output_dir: Path) -> None:
    """Download existing results from S3."""
    model_prefix = model_name.replace('/', '_')
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/"
    try:
        subprocess.run(
            ["aws", "s3", "sync", s3_path, str(output_dir), "--quiet"],
            check=False,
            capture_output=True,
        )
        print(f"Synced existing results from S3: {s3_path}")
    except Exception as e:
        print(f"S3 download skipped: {e}")


def s3_upload_file(local_path: Path, model_name: str) -> None:
    """Upload a single file to S3."""
    model_prefix = model_name.replace('/', '_')
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/{local_path.name}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(local_path), s3_path, "--quiet"],
            check=True,
            capture_output=True,
        )
        print(f"  Uploaded to S3: {s3_path}")
    except Exception as e:
        print(f"  S3 upload failed: {e}")


def load_categorized_benchmarks() -> Dict[str, List[str]]:
    """Load benchmarks grouped by category."""
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    with open(params_dir / "working_benchmarks_categorized.json") as f:
        return json.load(f)


def load_category_directions() -> Dict[str, Dict]:
    """Load hypothesized directions for each category."""
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    with open(params_dir / "category_directions.json") as f:
        return json.load(f)


@dataclass
class OptimalConfig:
    """Optimal configuration for extracting a concept."""
    # Best layer configuration
    optimal_layer: int  # Single best layer
    optimal_layer_range: List[int]  # Top 3 layers within 5% of best
    layer_accuracy: float  # Accuracy at optimal layer
    
    # Best extraction strategy
    optimal_strategy: str  # e.g., "last_token", "mean_diff"
    strategy_accuracy: float
    
    # Minimum pairs needed
    min_pairs_for_stable_signal: int  # Minimum n for accuracy within 5% of max
    pairs_saturation_curve: Dict[int, float]  # n_pairs -> accuracy
    
    # Steering configuration
    num_directions_needed: int  # k for multi-direction steering
    single_direction_accuracy: float  # accuracy with k=1
    multi_direction_accuracy: float  # accuracy with optimal k
    steering_gain: float  # multi - single direction accuracy
    
    # Signal quality
    icd: float  # Intrinsic Concept Dimensionality
    signal_above_noise: float  # accuracy - nonsense_accuracy
    is_linear: bool
    verdict: str  # STRONG_CONCEPT, DIFFUSE_CONCEPT, WEAK_SIGNAL, NO_SIGNAL


@dataclass
class CategoryResult:
    """Result for a single category."""
    category: str
    description: str
    benchmarks_tested: List[str]
    total_tests: int
    
    # Step 1: Signal detection
    avg_signal_strength: float  # MLP CV accuracy
    signal_exists: bool  # avg_signal_strength > 0.6
    
    # Step 2: Linearity check  
    avg_linear_probe_accuracy: float  # Linear probe CV accuracy
    is_linear: bool  # signal is linear (CAA will work)
    
    # NEW: Nonlinear signal metrics
    avg_knn_accuracy_k10: float  # k-NN CV accuracy
    avg_knn_pca_accuracy: float  # k-NN on PCA features (addresses curse of dimensionality)
    avg_knn_umap_accuracy: float  # k-NN on UMAP features (preserves nonlinear structure)
    avg_knn_pacmap_accuracy: float  # k-NN on PaCMAP features (preserves local+global structure)
    avg_mlp_probe_accuracy: float  # MLP probe (regularized nonlinear)
    avg_best_nonlinear: float  # max(knn, knn_pca, knn_umap, knn_pacmap, mlp) - best nonlinear signal detector
    avg_mmd_rbf: float  # Maximum Mean Discrepancy
    avg_local_dim_pos: float  # Local intrinsic dim of positive class
    avg_local_dim_neg: float  # Local intrinsic dim of negative class
    avg_fisher_max: float  # Max Fisher ratio
    avg_density_ratio: float  # Density ratio
    
    # Step 3: Geometry details (only meaningful if signal_exists)
    structure_distribution: Dict[str, int]
    structure_percentages: Dict[str, float]
    dominant_structure: str
    avg_linear_score: float
    avg_cohens_d: float
    
    # Multi-direction analysis: how many directions needed?
    avg_multi_dir_accuracy_k1: float  # accuracy with 1 direction
    avg_multi_dir_accuracy_k3: float  # accuracy with 3 directions
    avg_multi_dir_accuracy_k5: float  # accuracy with 5 directions
    avg_multi_dir_min_k: float  # average min k needed for good accuracy
    avg_multi_dir_gain: float  # average gain from using multiple directions
    
    # NEW: ICD (Intrinsic Concept Dimensionality) and nonsense baseline
    avg_icd: float  # average Intrinsic Concept Dimensionality
    avg_icd_top1_variance: float  # average variance explained by top-1 direction
    avg_nonsense_icd: float  # ICD of nonsense baseline (random tokens)
    avg_icd_ratio: float  # nonsense_icd / real_icd (higher = more concentrated)
    avg_nonsense_accuracy: float  # accuracy on random token baseline
    avg_signal_above_baseline: float  # real accuracy - nonsense accuracy
    signal_verdict: str  # STRONG_CONCEPT, DIFFUSE_CONCEPT, WEAK_SIGNAL, NO_SIGNAL
    
    # Final recommendation
    recommendation: str  # NO_SIGNAL, CAA, or NONLINEAR
    has_unified_direction: bool
    best_config: Optional[Dict[str, Any]] = None
    
    # NEW: Optimal configuration (full diagnosis)
    optimal_config: Optional[OptimalConfig] = None


@dataclass 
class DiscoveryResults:
    """Results from full discovery run."""
    model: str
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    
    def summary(self) -> str:
        lines = [
            f"Model: {self.model}",
            f"Categories analyzed: {len(self.categories)}",
            "",
        ]
        
        # Group by recommendation
        caa_ready = []  # Has signal AND linear
        nonlinear = []  # Has signal but NOT linear
        no_signal = []  # No signal
        
        for name, cat in self.categories.items():
            if not cat.signal_exists:
                no_signal.append(name)
            elif cat.is_linear:
                caa_ready.append(name)
            else:
                nonlinear.append(name)
        
        if caa_ready:
            lines.append(f"CAA READY - Linear signal ({len(caa_ready)}):")
            for name in sorted(caa_ready, key=lambda n: self.categories[n].avg_best_nonlinear, reverse=True):
                cat = self.categories[name]
                lines.append(f"  {name}: best_nonlinear={cat.avg_best_nonlinear:.2f}, linear={cat.avg_linear_probe_accuracy:.2f}, gap={cat.avg_linear_probe_accuracy - cat.avg_best_nonlinear:.2f}")
        
        if nonlinear:
            lines.append(f"\nNONLINEAR - Need different method ({len(nonlinear)}):")
            for name in nonlinear:
                cat = self.categories[name]
                lines.append(f"  {name}: best_nonlinear={cat.avg_best_nonlinear:.2f}, linear={cat.avg_linear_probe_accuracy:.2f}, gap={cat.avg_linear_probe_accuracy - cat.avg_best_nonlinear:.2f}")
        
        if no_signal:
            lines.append(f"\nNO SIGNAL ({len(no_signal)}):")
            for name in no_signal:
                cat = self.categories[name]
                lines.append(f"  {name}: best_nonlinear={cat.avg_best_nonlinear:.2f} (kNN={cat.avg_knn_accuracy_k10:.2f}, kNN_pca={cat.avg_knn_pca_accuracy:.2f}, kNN_umap={cat.avg_knn_umap_accuracy:.2f}, kNN_pacmap={cat.avg_knn_pacmap_accuracy:.2f}, MLP={cat.avg_mlp_probe_accuracy:.2f})")
        
        return "\n".join(lines)

