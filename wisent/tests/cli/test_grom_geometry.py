"""
Tests for GROM geometry adaptation integration.

Tests geometry detection, pre-flight checks, and GROM adaptive training.
"""

import argparse
import subprocess
import tempfile
import os
import json
import torch

from wisent.core.utils.config_tools.constants import (
    DEFAULT_RANDOM_SEED,
    SEPARATOR_WIDTH_STANDARD,
)

try:
    import pytest
except ImportError:
    pytest = None


def test_grom_geometry_on_truthfulqa(*, geometry_optimization_steps: int):
    """Test GROM geometry adaptation on truthfulqa_gen task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        activations_file = os.path.join(tmpdir, "activations.json")
        
        # Step 1: Generate pairs
        print("\n[Step 1] Generating truthfulqa_gen pairs...")
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.primitives.model_interface.core.main", "generate-pairs-from-task",
                "truthfulqa_gen",
                "--output", pairs_file,
                "--limit", "30",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        assert result.returncode == 0, f"Pair generation failed: {result.stderr}"
        assert os.path.exists(pairs_file), "Pairs file not created"
        
        # Step 2: Get activations
        print("[Step 2] Collecting activations...")
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.primitives.model_interface.core.main", "get-activations",
                pairs_file,
                "--output", activations_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layers", "8,12,15",
                "--token-aggregation", "average",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        assert result.returncode == 0, f"Activation collection failed: {result.stderr}"
        assert os.path.exists(activations_file), "Activations file not created"
        
        # Step 3: Run geometry analysis and GROM training
        print("[Step 3] Running geometry analysis and GROM training...")
        
        from wisent.core.primitives.contrastive_pairs.diagnostics import detect_geometry_structure
        from wisent.core.control.steering_methods.preflight import run_preflight_check
        from wisent.core.control.steering_methods.methods.grom import GROMMethod
        
        # Load activations from JSON
        with open(activations_file, 'r') as f:
            activations_data = json.load(f)
        
        pairs_list = activations_data.get('pairs', [])
        assert len(pairs_list) > 0, "No pairs loaded"
        
        # Extract activations for geometry analysis
        pos_acts = []
        neg_acts = []
        for pair in pairs_list:
            pos_la = pair.get('positive_response', {}).get('layers_activations', {})
            neg_la = pair.get('negative_response', {}).get('layers_activations', {})
            # Layer keys are just numbers like "15", not "layer_15"
            if "15" in pos_la and "15" in neg_la:
                pos_acts.append(torch.tensor(pos_la["15"]).reshape(-1))
                neg_acts.append(torch.tensor(neg_la["15"]).reshape(-1))
        
        assert len(pos_acts) > 0, "No activations extracted"
        
        pos_tensor = torch.stack(pos_acts)
        neg_tensor = torch.stack(neg_acts)
        
        # Geometry analysis
        geo_result = detect_geometry_structure(pos_tensor, neg_tensor)
        print(f"   Detected: {geo_result.best_structure.value} ({geo_result.best_score:.3f})")
        for name, score in geo_result.all_scores.items():
            print(f"     {name}: {score.score:.3f}")
        
        assert geo_result.best_score > 0, "Geometry detection failed"
        
        # Pre-flight checks
        print("[Step 4] Running pre-flight checks...")
        from wisent.core.control.steering_methods._helpers.preflight_helpers import PreflightThresholds
        _pf_thresholds = PreflightThresholds()
        caa_check = run_preflight_check(pos_tensor, neg_tensor, "caa", thresholds=_pf_thresholds)
        grom_check = run_preflight_check(pos_tensor, neg_tensor, "grom", thresholds=_pf_thresholds)
        
        print(f"   CAA: {caa_check.compatibility_score:.0%} compatible")
        print(f"   GROM: {grom_check.compatibility_score:.0%} compatible")
        
        assert caa_check.compatibility_score >= 0, "CAA pre-flight failed"
        assert grom_check.compatibility_score >= 0, "GROM pre-flight failed"
        
        # GROM geometry adaptation analysis (without full training)
        print("[Step 5] Testing GROM geometry adaptation logic...")
        method = GROMMethod(
            num_directions=5,
            steering_layers=[8, 12, 15],
            sensor_layer=15,
            adapt_to_geometry=True,
            optimization_steps=geometry_optimization_steps,
        )
        
        # Test the geometry adaptation logic directly with our tensors
        # Construct activation buckets in the format GROM expects
        hidden_dim = pos_tensor.shape[1]
        buckets = {
            'layer_15': (
                [t.unsqueeze(0) for t in pos_tensor],
                [t.unsqueeze(0) for t in neg_tensor]
            )
        }
        
        adaptation = method._analyze_and_adapt_geometry(buckets, ['layer_15'], hidden_dim)
        
        assert adaptation is not None, "Geometry adaptation failed"
        print(f"\n   GEOMETRY ADAPTATION RESULTS:")
        print(f"     Detected: {adaptation.detected_structure}")
        print(f"     Directions: {adaptation.original_num_directions} -> {adaptation.adapted_num_directions}")
        print(f"     Gating: {adaptation.gating_enabled}")
        for adapt in adaptation.adaptations_made:
            print(f"     - {adapt}")
        
        print("\n" + "=" * SEPARATOR_WIDTH_STANDARD)
        print("TEST PASSED - GROM geometry adaptation working!")
        print("=" * SEPARATOR_WIDTH_STANDARD)


def test_geometry_detection_synthetic(*, hidden_dim: int, n_samples: int):
    """Test geometry detection with synthetic data."""
    from wisent.core.primitives.contrastive_pairs.diagnostics import detect_geometry_structure

    torch.manual_seed(DEFAULT_RANDOM_SEED)

    # Create linear data
    base = torch.randn(hidden_dim)
    base = base / base.norm()
    pos_linear = base.unsqueeze(0) * 3 + torch.randn(n_samples, hidden_dim) * 0.05
    neg_linear = base.unsqueeze(0) * -3 + torch.randn(n_samples, hidden_dim) * 0.05

    result = detect_geometry_structure(pos_linear, neg_linear)
    
    assert result.best_structure.value == "linear", f"Expected linear, got {result.best_structure.value}"
    assert result.all_scores["linear"].score > 0.9, f"Linear score too low: {result.all_scores['linear'].score}"
    print(f"Linear detection: {result.all_scores['linear'].score:.3f}")


def test_preflight_checks(*, hidden_dim: int, n_samples: int):
    """Test pre-flight check system."""
    from wisent.core.control.steering_methods.preflight import run_preflight_check

    torch.manual_seed(DEFAULT_RANDOM_SEED)

    # Linear data
    base = torch.randn(hidden_dim)
    base = base / base.norm()
    pos = base.unsqueeze(0) * 3 + torch.randn(n_samples, hidden_dim) * 0.05
    neg = base.unsqueeze(0) * -3 + torch.randn(n_samples, hidden_dim) * 0.05

    from wisent.core.control.steering_methods._helpers.preflight_helpers import PreflightThresholds
    _pf_thresholds = PreflightThresholds()
    caa_result = run_preflight_check(pos, neg, "caa", thresholds=_pf_thresholds)
    grom_result = run_preflight_check(pos, neg, "grom", thresholds=_pf_thresholds)
    
    # For linear data, CAA should be highly compatible
    assert caa_result.compatibility_score > 0.8, f"CAA should be compatible for linear: {caa_result.compatibility_score}"
    assert grom_result.compatibility_score >= 0, "GROM check failed"
    
    print(f"CAA compatibility: {caa_result.compatibility_score:.0%}")
    print(f"GROM compatibility: {grom_result.compatibility_score:.0%}")


def test_grom_adaptation_linear(*, hidden_dim: int, n_samples: int):
    """Test GROM adapts correctly to linear data."""
    from wisent.core.control.steering_methods.methods.grom import GROMMethod

    torch.manual_seed(DEFAULT_RANDOM_SEED)

    # Create linear data
    base = torch.randn(hidden_dim)
    base = base / base.norm()
    pos_linear = base.unsqueeze(0) * 3 + torch.randn(n_samples, hidden_dim) * 0.05
    neg_linear = base.unsqueeze(0) * -3 + torch.randn(n_samples, hidden_dim) * 0.05

    method = GROMMethod(
        num_directions=5,
        steering_layers=[0],
        sensor_layer=0,
        adapt_to_geometry=True,
        linear_threshold=0.8,
    )
    
    # Test adaptation logic directly
    buckets = {'layer_0': (
        [t.unsqueeze(0) for t in pos_linear],
        [t.unsqueeze(0) for t in neg_linear]
    )}
    
    adaptation = method._analyze_and_adapt_geometry(buckets, ['layer_0'], hidden_dim)
    
    assert adaptation.detected_structure == "linear", f"Expected linear, got {adaptation.detected_structure}"
    assert adaptation.adapted_num_directions == 1, f"Expected 1 direction for linear, got {adaptation.adapted_num_directions}"
    assert adaptation.gating_enabled == False, "Gating should be disabled for linear"
    
    print(f"Detected: {adaptation.detected_structure}")
    print(f"Directions: {adaptation.original_num_directions} -> {adaptation.adapted_num_directions}")
    print(f"Gating: {adaptation.gating_enabled}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GROM geometry tests")
    parser.add_argument("--hidden-dim", type=int, required=True,
                        help="Hidden dimension for synthetic test data")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Number of synthetic samples for test data")
    parser.add_argument("--geometry-optimization-steps", type=int, required=True,
                        help="Optimization steps for geometry detection")
    cli_args = parser.parse_args()

    test_kwargs = dict(hidden_dim=cli_args.hidden_dim, n_samples=cli_args.n_samples)

    print("Running GROM geometry tests...\n")

    for label, fn in [
        ("Geometry detection with synthetic data", test_geometry_detection_synthetic),
        ("Pre-flight checks", test_preflight_checks),
        ("GROM adaptation for linear data", test_grom_adaptation_linear),
    ]:
        print("=" * SEPARATOR_WIDTH_STANDARD)
        print(label)
        print("=" * SEPARATOR_WIDTH_STANDARD)
        fn(**test_kwargs)
        print("PASSED\n")

    print("=" * SEPARATOR_WIDTH_STANDARD)
    print("Full GROM geometry on truthfulqa_gen")
    print("=" * SEPARATOR_WIDTH_STANDARD)
    test_grom_geometry_on_truthfulqa(geometry_optimization_steps=cli_args.geometry_optimization_steps)
    print("PASSED\n")

    print("\nAll tests passed!")
