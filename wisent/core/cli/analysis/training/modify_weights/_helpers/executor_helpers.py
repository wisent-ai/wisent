"""Executor helper functions for szlak, wicher modes and diagnostics saving."""

from __future__ import annotations

import json


def execute_szlak_mode(args, model, tokenizer, wisent_model, pairs):
    """Execute SZLAK weight modification mode."""
    from wisent.core.cli.analysis.training.modify_weights.method_training import train_szlak_for_task
    from wisent.core.weight_modification.export import export_szlak_model

    if args.verbose:
        print("Training SZLAK for geodesic OT steering...")

    szlak_result = train_szlak_for_task(args, wisent_model, pairs)

    if args.verbose:
        print(f"\nExporting SZLAK model...")

    export_szlak_model(
        model=model, szlak_steering=szlak_result, save_path=args.output_dir,
        tokenizer=tokenizer, base_strength=args.strength,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )

    if args.verbose:
        print(f"\nSZLAK model exported to {args.output_dir}")
        print(f"  Layers: {len(szlak_result.source_points)}")


def execute_wicher_mode(args, model, tokenizer, wisent_model, pairs):
    """Execute WICHER weight modification mode."""
    from wisent.core.cli.analysis.training.modify_weights.method_training import train_wicher_for_task
    from wisent.core.weight_modification.export import export_wicher_model

    if args.verbose:
        print("Training WICHER for Broyden-based steering...")

    wicher_result = train_wicher_for_task(args, wisent_model, pairs)

    if args.verbose:
        print(f"\nExporting WICHER model...")

    export_wicher_model(
        model=model, wicher_steering=wicher_result, save_path=args.output_dir,
        tokenizer=tokenizer, base_strength=args.strength,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )

    if args.verbose:
        print(f"\nWICHER model exported to {args.output_dir}")
        print(f"  Layers: {len(wicher_result.concept_directions)}")


def save_diagnostics(path: str, result):
    """Save diagnostics to file."""
    diagnostics_data = {
        "layers": {str(layer): {"linear_score": diag.linear_score, "fisher_ratio": diag.fisher_ratio}
                   for layer, diag in result.layer_diagnostics.items()},
        "layer_weights": {str(k): v for k, v in result.layer_weights.items()},
        "mode_used": result.mode_used.value,
    }
    with open(path, 'w') as f:
        json.dump(diagnostics_data, f, indent=2)
