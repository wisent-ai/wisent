"""CLI command for steering effect visualization."""

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

    # Load all pair texts
    all_pair_texts = load_pair_texts_from_database(
        task_name=args.task, limit=1000, database_url=args.database_url
    )
    print(f"Loaded {len(all_pair_texts)} pair texts")

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

        # Get pairs for this concept
        concept_pair_ids = [pid for pid, cid in pair_assignments.items() if cid == concept_id]
        concept_pair_ids = [pid for pid in concept_pair_ids if pid in all_pair_texts]
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
            responses.append({"pair_id": pid, "prompt": prompt, "base": base_resp, "base_eval": base_result.ground_truth, "steered": steered_resp, "steered_eval": steered_result.ground_truth})

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


def execute_steering_viz(args):
    """Execute the steering-viz command."""
    import sys
    print("DEBUG: Starting steering-viz...", flush=True)

    import torch
    print("DEBUG: Imported torch", flush=True)

    import random
    print("DEBUG: Importing database loaders...", flush=True)

    from wisent.core.geometry.repscan_with_concepts import (
        load_activations_from_database,
        load_pair_texts_from_database,
    )
    print("DEBUG: Imported database loaders", flush=True)

    from wisent.core.geometry.steering_visualizations import create_steering_effect_figure
    from wisent.core.wisent import Wisent
    print("DEBUG: All imports complete", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEERING EFFECT VISUALIZATION", flush=True)
    print(f"{'='*60}", flush=True)

    # Load ALL pair texts first for 80/20 split
    print(f"\nLoading all pairs for train/test split...", flush=True)
    all_pair_texts = load_pair_texts_from_database(
        task_name=args.task,
        limit=1000,  # Load full truthfulqa_custom (~817 pairs)
        database_url=args.database_url,
    )
    all_pair_ids = list(all_pair_texts.keys())
    random.seed(42)
    random.shuffle(all_pair_ids)

    split_idx = int(len(all_pair_ids) * 0.8)
    train_ids = set(all_pair_ids[:split_idx])
    test_ids = all_pair_ids[split_idx:]

    print(f"  Total pairs: {len(all_pair_ids)}")
    print(f"  Train (80%): {len(train_ids)}")
    print(f"  Test (20%): {len(test_ids)}")

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
        pair_ids=train_ids,  # Only use train set for steering vector
    )
    print(f"  Loaded {len(pos_ref)} training reference pairs", flush=True)

    # Compute steering vector (mean difference)
    steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
    print(f"  Steering vector norm: {steering_vector.norm().item():.4f}", flush=True)

    # Use test set for evaluation
    pair_texts = {pid: all_pair_texts[pid] for pid in test_ids}
    test_prompts = [pair_texts[pid].get("prompt", "") for pid in test_ids]
    print(f"\nTest set: {len(test_prompts)} prompts", flush=True)

    # Load model
    print(f"\nLoading model: {args.model}", flush=True)
    wisent = Wisent.for_text(args.model)

    # Extract base and steered activations
    print(f"\nExtracting base and steered activations...", flush=True)
    adapter = wisent.adapter
    layer_name = f"layer.{args.layer}"

    base_acts = []
    steered_acts = []

    for i, prompt in enumerate(test_prompts):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(test_prompts)}...", flush=True)

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

    print(f"  Extracted {len(base_activations)} base/steered pairs", flush=True)

    # Evaluate responses using native evaluator
    print(f"\nEvaluating responses...", flush=True)
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.adapters.base import SteeringConfig

    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator_rotator = EvaluatorRotator(evaluator=None, task_name=args.task)
    evaluator = evaluator_rotator.current
    print(f"  Using evaluator: {evaluator.name}", flush=True)

    # Create steering vectors for generation
    steering_vectors = LayerActivations({layer_name: args.strength * steering_vector})
    max_new_tokens = getattr(args, 'max_new_tokens', 100)

    base_evaluations = []
    steered_evaluations = []
    all_responses = []  # Store all responses for JSON output
    sample_keys = test_ids  # Use all test IDs

    for i, pair_key in enumerate(sample_keys):
        pair_data = pair_texts[pair_key]
        prompt = pair_data.get("prompt", "")
        positive_reference = pair_data.get("positive", "")
        negative_reference = pair_data.get("negative", "")
        correct_answers = [positive_reference] if positive_reference else []
        incorrect_answers = [negative_reference] if negative_reference else []

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = adapter.apply_chat_template(messages, add_generation_prompt=True)

        # Generate base response (unsteered)
        base_response_raw = adapter._generate_unsteered(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
        )
        # Strip prompt from response (find assistant marker)
        if "assistant\n\n" in base_response_raw:
            base_response = base_response_raw.split("assistant\n\n", 1)[-1].strip()
        elif "assistant\n" in base_response_raw:
            base_response = base_response_raw.split("assistant\n", 1)[-1].strip()
        else:
            base_response = base_response_raw

        # Generate steered response
        steered_response_raw = adapter.forward_with_steering(
            formatted_prompt,
            steering_vectors=steering_vectors,
            config=SteeringConfig(scale=1.0),
        )
        if "assistant\n\n" in steered_response_raw:
            steered_response = steered_response_raw.split("assistant\n\n", 1)[-1].strip()
        elif "assistant\n" in steered_response_raw:
            steered_response = steered_response_raw.split("assistant\n", 1)[-1].strip()
        else:
            steered_response = steered_response_raw

        # Evaluate both (compare to correct AND incorrect answers)
        base_result = evaluator.evaluate(base_response, positive_reference, correct_answers=correct_answers, incorrect_answers=incorrect_answers)
        steered_result = evaluator.evaluate(steered_response, positive_reference, correct_answers=correct_answers, incorrect_answers=incorrect_answers)

        base_evaluations.append(base_result.ground_truth)
        steered_evaluations.append(steered_result.ground_truth)

        # Store response data
        all_responses.append({
            "pair_id": pair_key,
            "prompt": prompt,
            "positive_reference": positive_reference,
            "negative_reference": negative_reference,
            "base_response": base_response,
            "base_eval": base_result.ground_truth,
            "steered_response": steered_response,
            "steered_eval": steered_result.ground_truth,
        })

        if i % 10 == 0:
            print(f"  Evaluated {i+1}/{len(sample_keys)}...", flush=True)

    base_truthful = sum(1 for e in base_evaluations if e == "TRUTHFUL")
    steered_truthful = sum(1 for e in steered_evaluations if e == "TRUTHFUL")
    print(f"  Base: {base_truthful}/{len(base_evaluations)} TRUTHFUL", flush=True)
    print(f"  Steered: {steered_truthful}/{len(steered_evaluations)} TRUTHFUL", flush=True)

    # Generate visualization with evaluations
    print(f"\nGenerating visualization...", flush=True)
    viz_b64 = create_steering_effect_figure(
        pos_activations=pos_ref,
        neg_activations=neg_ref,
        base_activations=base_activations,
        steered_activations=steered_activations,
        title=f"Steering Effect: {args.task} (layer {args.layer}, strength {args.strength})",
        base_evaluations=base_evaluations,
        steered_evaluations=steered_evaluations,
    )

    # Save visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = base64.b64decode(viz_b64)
    with open(output_path, 'wb') as f:
        f.write(png_bytes)

    print(f"\nVisualization saved to: {output_path}")

    # Save responses to JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            "model": args.model,
            "task": args.task,
            "layer": args.layer,
            "strength": args.strength,
            "summary": {
                "base_truthful": base_truthful,
                "steered_truthful": steered_truthful,
                "total": len(base_evaluations),
            },
            "responses": all_responses,
        }, f, indent=2)
    print(f"Responses saved to: {json_path}", flush=True)

    print(f"\n{'='*60}")
    print("STEERING VISUALIZATION COMPLETE")
    print(f"{'='*60}")

    return {"output": str(output_path), "json_output": str(json_path)}
