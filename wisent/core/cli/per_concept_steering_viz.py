"""CLI command for per-concept steering effect visualization."""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import json
import base64
from pathlib import Path


def execute_per_concept_steering_viz(args):
    """Execute per-concept steering visualization with evaluation."""
    import torch
    import random
    from wisent.core.geometry.repscan_with_concepts import (
        load_activations_from_database,
        load_pair_texts_from_database,
    )
    from wisent.core.geometry.steering_visualizations import create_per_concept_steering_figure
    from wisent.core.wisent import Wisent

    print(f"\n{'='*60}")
    print("PER-CONCEPT STEERING VISUALIZATION WITH EVALUATION")
    print(f"{'='*60}")

    # Load repscan results
    if not args.repscan_results:
        print("ERROR: --repscan-results is required for --per-concept")
        sys.exit(1)

    with open(args.repscan_results) as f:
        repscan = json.load(f)

    concepts = repscan["concept_decomposition"]["concepts"]
    pair_assignments = repscan["concept_decomposition"]["pair_assignments"]
    print(f"Loaded {len(concepts)} concepts from repscan results")

    # Load pair texts - prefer stored pair_texts from repscan results
    stored_pair_texts = repscan["concept_decomposition"].get("pair_texts")
    if stored_pair_texts:
        all_pair_texts = {int(k) if isinstance(k, str) else k: v for k, v in stored_pair_texts.items()}
        print(f"Loaded {len(all_pair_texts)} pair texts from repscan results")
    else:
        all_pair_texts = load_pair_texts_from_database(
            task_name=args.task, limit=1000, database_url=args.database_url
        )
        print(f"Loaded {len(all_pair_texts)} pair texts from database")

    # Load model
    print(f"\nLoading model: {args.model}")
    wisent = Wisent.for_text(args.model)
    adapter = wisent.adapter

    # Setup evaluator
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.adapters.base import SteeringConfig

    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task).current
    print(f"Using evaluator: {evaluator.name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for concept in concepts:
        concept_id = concept["id"]
        concept_name = concept.get("name", f"concept_{concept_id}")
        optimal_layer = concept.get("optimal_layer", args.layer)
        print(f"\n{'='*40}")
        print(f"Concept {concept_id}: {concept_name}")
        print(f"  Optimal layer: {optimal_layer}")

        # Get pairs for this concept - handle both string and int keys
        concept_pair_ids = []
        for pid, cid in pair_assignments.items():
            if cid == concept_id:
                pid_int = int(pid) if isinstance(pid, str) else pid
                if pid_int in all_pair_texts:
                    concept_pair_ids.append(pid_int)
                elif str(pid) in all_pair_texts:
                    concept_pair_ids.append(str(pid))
        print(f"  Pairs in concept: {len(concept_pair_ids)}")

        if len(concept_pair_ids) < 5:
            print(f"  Skipping - not enough pairs")
            continue

        # Split 80/20
        random.seed(42)
        random.shuffle(concept_pair_ids)
        split_idx = int(len(concept_pair_ids) * 0.8)
        train_ids = set(concept_pair_ids[:split_idx])
        test_ids = concept_pair_ids[split_idx:]
        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

        if len(test_ids) < 2:
            print(f"  Skipping - not enough test pairs")
            continue

        # Load reference activations for this concept
        pos_ref, neg_ref = load_activations_from_database(
            model_name=args.model, task_name=args.task, layer=optimal_layer,
            prompt_format=args.prompt_format, extraction_strategy=args.extraction_strategy,
            limit=500, database_url=args.database_url, pair_ids=train_ids
        )
        print(f"  Loaded {len(pos_ref)} training reference pairs")

        if len(pos_ref) == 0:
            print(f"  Skipping - no reference activations loaded")
            continue

        # Compute concept-specific steering vector
        steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
        layer_name = f"layer.{optimal_layer}"
        steering_vectors = LayerActivations({layer_name: args.strength * steering_vector})

        # Run test prompts
        base_acts, steered_acts = [], []
        base_evals, steered_evals = [], []
        responses = []

        for pid in test_ids:
            pair = all_pair_texts[pid]
            prompt = pair.get("prompt", "")
            pos_ref_text = pair.get("positive", "")
            neg_ref_text = pair.get("negative", "")

            messages = [{"role": "user", "content": prompt}]
            formatted = adapter.apply_chat_template(messages, add_generation_prompt=True)

            # Base activation
            base_layer_acts = adapter.extract_activations(prompt, layers=[layer_name])
            base_act = base_layer_acts.get(layer_name)
            if base_act is None:
                continue
            base_acts.append(base_act[0, -1, :].cpu())
            steered_act = base_act[0, -1, :] + args.strength * steering_vector.to(base_act.device)
            steered_acts.append(steered_act.cpu())

            # Generate responses
            base_resp = adapter._generate_unsteered(formatted, max_new_tokens=100, temperature=0.1, do_sample=True)
            if "assistant\n" in base_resp:
                base_resp = base_resp.split("assistant\n")[-1].strip()
            steered_resp = adapter.forward_with_steering(formatted, steering_vectors=steering_vectors, config=SteeringConfig(scale=1.0))
            if "assistant\n" in steered_resp:
                steered_resp = steered_resp.split("assistant\n")[-1].strip()

            # Evaluate
            base_result = evaluator.evaluate(base_resp, pos_ref_text, correct_answers=[pos_ref_text], incorrect_answers=[neg_ref_text])
            steered_result = evaluator.evaluate(steered_resp, pos_ref_text, correct_answers=[pos_ref_text], incorrect_answers=[neg_ref_text])
            base_evals.append(base_result.ground_truth)
            steered_evals.append(steered_result.ground_truth)
            responses.append({"pair_id": str(pid), "prompt": prompt, "base": base_resp, "base_eval": base_result.ground_truth, "steered": steered_resp, "steered_eval": steered_result.ground_truth})

        if not base_acts:
            print(f"  No activations extracted")
            continue

        base_activations = torch.stack(base_acts)
        steered_activations = torch.stack(steered_acts)

        base_truthful = sum(1 for e in base_evals if e == "TRUTHFUL")
        steered_truthful = sum(1 for e in steered_evals if e == "TRUTHFUL")
        print(f"  Base: {base_truthful}/{len(base_evals)} TRUTHFUL")
        print(f"  Steered: {steered_truthful}/{len(steered_evals)} TRUTHFUL")

        # Create visualization
        viz_b64 = create_per_concept_steering_figure(
            concept_name=concept_name, concept_id=concept_id,
            pos_activations=pos_ref, neg_activations=neg_ref,
            base_activations=base_activations, steered_activations=steered_activations,
            base_evaluations=base_evals, steered_evaluations=steered_evals,
            layer=optimal_layer, strength=args.strength
        )

        # Save
        png_path = output_dir / f"concept_{concept_id}_{concept_name}.png"
        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(viz_b64))
        print(f"  Saved: {png_path}")

        all_results.append({"concept_id": concept_id, "name": concept_name, "layer": optimal_layer, "base_truthful": base_truthful, "steered_truthful": steered_truthful, "total": len(base_evals), "responses": responses})

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print(f"\n{'='*60}")
    print("PER-CONCEPT STEERING VISUALIZATION COMPLETE")
    print(f"{'='*60}")
