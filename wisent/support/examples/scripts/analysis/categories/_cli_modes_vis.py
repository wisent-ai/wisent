"""CLI mode handlers for visualization and single-sample test."""

import random
from pathlib import Path

from wisent.core.utils.config_tools.constants import SEPARATOR_WIDTH_WIDE
from wisent.core.primitives.models.wisent_model import WisentModel

from ._data_loading import (
    load_truthfulqa_pairs,
    load_hellaswag_pairs,
    extract_difference_vectors,
)
from ._visualization_advanced import visualize_concept_detection
from ._orchestration import run_single_sample_detection


def run_visualize_mode(args):
    """Run visualization mode."""
    # Run visualization mode
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("CONCEPT DETECTION VISUALIZATION")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    vis_output = Path(args.vis_output_dir)
    vis_output.mkdir(parents=True, exist_ok=True)
    
    # Load pairs
    print("\nLoading TruthfulQA pairs...")
    tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(tqa_pairs)} pairs")
    
    print("Loading HellaSwag pairs...")
    hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(hs_pairs)} pairs")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = WisentModel(args.model, device="mps")
    
    layer = args.layer if args.layer else model.num_layers // 2
    print(f"Using layer: {layer}")
    
    # Create mixed sample
    mixed_pairs = tqa_pairs + hs_pairs
    random.seed(args.seed)
    random.shuffle(mixed_pairs)
    
    # Extract activations for all conditions
    print("\n--- Extracting activations ---")
    
    print("\nMixed sample...")
    mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
    
    print("\nTruthfulQA sample...")
    tqa_diffs, tqa_sources = extract_difference_vectors(model, tqa_pairs, layer)
    
    print("\nHellaSwag sample...")
    hs_diffs, hs_sources = extract_difference_vectors(model, hs_pairs, layer)
    
    # Generate visualizations
    print("\n--- Generating visualizations ---")
    
    print("\nMixed (TruthfulQA + HellaSwag):")
    visualize_concept_detection(
        mixed_diffs, mixed_sources,
        title="MIXED: TruthfulQA + HellaSwag (Should detect 2 concepts)",
        output_path=str(vis_output / "mixed_concepts.png"),
        show_plot=False
    )
    
    print("\nPure TruthfulQA:")
    visualize_concept_detection(
        tqa_diffs, tqa_sources,
        title="PURE: TruthfulQA Only (Should detect 1 concept)",
        output_path=str(vis_output / "truthfulqa_only.png"),
        show_plot=False
    )
    
    print("\nPure HellaSwag:")
    visualize_concept_detection(
        hs_diffs, hs_sources,
        title="PURE: HellaSwag Only (Should detect 1 concept)",
        output_path=str(vis_output / "hellaswag_only.png"),
        show_plot=False
    )
    
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"Visualizations saved to: {vis_output}")
    print("=" * SEPARATOR_WIDTH_WIDE)


def run_single_sample_test_mode(args):
    """Test single-sample detection on mixed vs pure samples."""
    # Test the single-sample detection on both mixed and pure samples
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("TESTING SINGLE-SAMPLE DETECTION")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("\nThis test verifies that single-sample detection works by testing")
    print("it on samples we KNOW are mixed vs pure.\n")
    
    # Load pairs
    print("Loading TruthfulQA pairs...")
    tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(tqa_pairs)} pairs")
    
    print("Loading HellaSwag pairs...")
    hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(hs_pairs)} pairs")
    
    # Create mixed sample
    mixed_pairs = tqa_pairs + hs_pairs
    random.seed(args.seed)
    random.shuffle(mixed_pairs)
    
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("TEST A: Mixed sample (should detect MULTIPLE_CONCEPTS)")
    print("=" * SEPARATOR_WIDTH_WIDE)
    mixed_result = run_single_sample_detection(
        args.model, mixed_pairs, args.layer, args.n_bootstrap, args.seed
    )
    
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("TEST B: Pure TruthfulQA (should detect SINGLE_CONCEPT)")
    print("=" * SEPARATOR_WIDTH_WIDE)
    tqa_result = run_single_sample_detection(
        args.model, tqa_pairs, args.layer, args.n_bootstrap, args.seed
    )
    
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("TEST C: Pure HellaSwag (should detect SINGLE_CONCEPT)")
    print("=" * SEPARATOR_WIDTH_WIDE)
    hs_result = run_single_sample_detection(
        args.model, hs_pairs, args.layer, args.n_bootstrap, args.seed
    )
    
    # Summary
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("SINGLE-SAMPLE DETECTION SUMMARY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"\n{'Sample':<20} {'Verdict':<20} {'Expected':<20} {'Match':<10}")
    print("-" * SEPARATOR_WIDTH_WIDE)
    
    tests = [
        ("Mixed", mixed_result["verdict"], "MULTIPLE_CONCEPTS"),
        ("TruthfulQA", tqa_result["verdict"], "SINGLE_CONCEPT"),
        ("HellaSwag", hs_result["verdict"], "SINGLE_CONCEPT"),
    ]
    
    for name, verdict, expected in tests:
        # POSSIBLY_MULTIPLE is acceptable for SINGLE_CONCEPT expectation
        if expected == "SINGLE_CONCEPT":
            match = "YES" if verdict in ["SINGLE_CONCEPT", "POSSIBLY_MULTIPLE"] else "NO"
        else:
            match = "YES" if verdict == expected else "PARTIAL" if verdict == "POSSIBLY_MULTIPLE" else "NO"
        print(f"{name:<20} {verdict:<20} {expected:<20} {match:<10}")
