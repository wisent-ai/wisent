"""CLI command for steering effect visualization."""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import base64
from pathlib import Path


def execute_steering_viz(args):
    """Execute the steering-viz command."""
    import torch
    from wisent.core.geometry.repscan_with_concepts import (
        load_activations_from_database,
        load_pair_texts_from_database,
    )
    from wisent.core.geometry.steering_visualizations import create_steering_effect_figure
    from wisent.core.wisent import Wisent

    print(f"\n{'='*60}")
    print("STEERING EFFECT VISUALIZATION")
    print(f"{'='*60}")

    # Load reference activations from database
    print(f"\nLoading reference activations...")
    print(f"  Model: {args.model}")
    print(f"  Task: {args.task}")
    print(f"  Layer: {args.layer}")

    pos_ref, neg_ref = load_activations_from_database(
        model_name=args.model,
        task_name=args.task,
        layer=args.layer,
        prompt_format=args.prompt_format,
        extraction_strategy=args.extraction_strategy,
        limit=args.limit,
        database_url=args.database_url,
    )
    print(f"  Loaded {len(pos_ref)} reference pairs")

    # Compute steering vector (mean difference)
    steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
    print(f"  Steering vector norm: {steering_vector.norm().item():.4f}")

    # Load pair texts to get prompts
    print(f"\nLoading test prompts...")
    pair_texts = load_pair_texts_from_database(
        task_name=args.task,
        limit=args.n_test_prompts,
        database_url=args.database_url,
    )
    test_prompts = [p.get("prompt", "") for p in pair_texts.values()][:args.n_test_prompts]
    print(f"  Loaded {len(test_prompts)} test prompts")

    # Load model
    print(f"\nLoading model: {args.model}")
    wisent = Wisent.for_text(args.model)

    # Extract base and steered activations
    print(f"\nExtracting base and steered activations...")
    adapter = wisent.adapter
    layer_name = f"layer.{args.layer}"

    base_acts = []
    steered_acts = []

    for i, prompt in enumerate(test_prompts):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(test_prompts)}...")

        # Base activation
        base_layer_acts = adapter.extract_activations(prompt, layers=[layer_name])
        base_act = base_layer_acts.get(layer_name)
        if base_act is not None:
            base_acts.append(base_act[0, -1, :])

        # Steered activation (add steering vector to base)
        if base_act is not None:
            steered_act = base_act[0, -1, :] + args.strength * steering_vector.to(base_act.device)
            steered_acts.append(steered_act.cpu())

    if not base_acts:
        print("ERROR: No activations extracted")
        sys.exit(1)

    base_activations = torch.stack(base_acts)
    steered_activations = torch.stack(steered_acts)

    print(f"  Extracted {len(base_activations)} base/steered pairs")

    # Generate visualization
    print(f"\nGenerating visualization...")
    viz_b64 = create_steering_effect_figure(
        pos_activations=pos_ref,
        neg_activations=neg_ref,
        base_activations=base_activations,
        steered_activations=steered_activations,
        title=f"Steering Effect: {args.task} (layer {args.layer}, strength {args.strength})",
    )

    # Save visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = base64.b64decode(viz_b64)
    with open(output_path, 'wb') as f:
        f.write(png_bytes)

    print(f"\nVisualization saved to: {output_path}")
    print(f"\n{'='*60}")
    print("STEERING VISUALIZATION COMPLETE")
    print(f"{'='*60}")

    return {"output": str(output_path)}
