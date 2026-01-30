"""CLI command for steering direction discovery."""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import random
import numpy as np
from pathlib import Path


def execute_discover_steering(args):
    """Execute the discover-steering command - find optimal steering directions."""
    import torch
    from wisent.core.geometry.repscan_with_concepts import (
        load_activations_from_database, load_pair_texts_from_database,
    )
    from wisent.core.geometry.steering_discovery import (
        discover_behavioral_direction, generate_candidate_directions,
        extract_generation_activations, compute_generation_direction, compare_directions,
    )
    from wisent.core.wisent import Wisent
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.adapters.base import SteeringConfig

    print(f"\n{'='*60}")
    print("STEERING DIRECTION DISCOVERY")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading data...")
    all_pair_texts = load_pair_texts_from_database(
        task_name=args.task, limit=1000, database_url=args.database_url
    )
    all_pair_ids = list(all_pair_texts.keys())
    random.seed(42)
    random.shuffle(all_pair_ids)

    split_idx = int(len(all_pair_ids) * 0.8)
    train_ids = set(all_pair_ids[:split_idx])
    test_ids = all_pair_ids[split_idx:][:args.n_test_samples]

    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Load reference activations
    pos_ref, neg_ref = load_activations_from_database(
        model_name=args.model, task_name=args.task, layer=args.layer,
        limit=500, database_url=args.database_url, pair_ids=train_ids
    )
    pos_np = pos_ref.cpu().numpy()
    neg_np = neg_ref.cpu().numpy()
    print(f"  Loaded {len(pos_ref)} reference pairs")

    # Original steering vector
    original_direction = pos_np.mean(axis=0) - neg_np.mean(axis=0)
    original_direction = original_direction / (np.linalg.norm(original_direction) + 1e-8)

    # Load model and evaluator
    print(f"\nLoading model: {args.model}")
    wisent = Wisent.for_text(args.model)
    adapter = wisent.adapter

    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task).current
    print(f"  Using evaluator: {evaluator.name}")

    # Get test prompts and references
    test_prompts = [all_pair_texts[pid].get("prompt", "") for pid in test_ids]
    test_pos_refs = [all_pair_texts[pid].get("positive", "") for pid in test_ids]
    test_neg_refs = [all_pair_texts[pid].get("negative", "") for pid in test_ids]

    layer_name = f"layer.{args.layer}"
    results = {"model": args.model, "task": args.task, "layer": args.layer, "methods": {}}

    def evaluate_direction(direction, strength=1.0):
        steering_vec = torch.from_numpy(direction).float() * strength
        steering_vectors = LayerActivations({layer_name: steering_vec})
        base_evals, steered_evals, base_resps, steered_resps = [], [], [], []
        for prompt, pos_ref, neg_ref in zip(test_prompts, test_pos_refs, test_neg_refs):
            msgs = [{"role": "user", "content": prompt}]
            fmt = adapter.apply_chat_template(msgs, add_generation_prompt=True)
            base_r = _extract_response(adapter._generate_unsteered(fmt, max_new_tokens=100, temperature=0.1))
            base_evals.append(evaluator.evaluate(base_r, pos_ref, correct_answers=[pos_ref], incorrect_answers=[neg_ref]).ground_truth)
            base_resps.append(base_r)
            steer_r = _extract_response(adapter.forward_with_steering(fmt, steering_vectors=steering_vectors, config=SteeringConfig(scale=1.0)))
            steered_evals.append(evaluator.evaluate(steer_r, pos_ref, correct_answers=[pos_ref], incorrect_answers=[neg_ref]).ground_truth)
            steered_resps.append(steer_r)
        base_t = sum(1 for e in base_evals if e == "TRUTHFUL")
        steered_t = sum(1 for e in steered_evals if e == "TRUTHFUL")
        return base_t, steered_t, base_evals, steered_evals, base_resps, steered_resps

    # 1. Baseline
    print(f"\n[1/4] BASELINE (mean diff)...")
    base_t, steered_t, base_evals, steer_evals, base_resps, _ = evaluate_direction(original_direction, args.strength)
    print(f"  Base: {base_t}/{len(test_ids)}, Steered: {steered_t}/{len(test_ids)}, Imp: {steered_t - base_t}")
    results["methods"]["baseline"] = {"base": base_t, "steered": steered_t, "improvement": steered_t - base_t}

    # Extract base activations
    base_acts = []
    for prompt in test_prompts:
        msgs = [{"role": "user", "content": prompt}]
        fmt = adapter.apply_chat_template(msgs, add_generation_prompt=True)
        acts = adapter.extract_activations(fmt, layers=[layer_name])
        if acts.get(layer_name) is not None:
            base_acts.append(acts[layer_name][0, -1, :].cpu().numpy())
    base_acts = np.array(base_acts)

    # 2. Behavioral Probing
    print(f"\n[2/4] BEHAVIORAL PROBING...")
    beh_dir, beh_det = discover_behavioral_direction(base_acts, base_evals)
    cos_sim = np.dot(original_direction, beh_dir)
    print(f"  Classifier acc: {beh_det['accuracy']:.3f}, Sim to baseline: {cos_sim:.3f}")
    b2, s2, _, _, _, _ = evaluate_direction(beh_dir, args.strength)
    print(f"  Base: {b2}/{len(test_ids)}, Steered: {s2}/{len(test_ids)}, Imp: {s2 - b2}")
    results["methods"]["behavioral"] = {"base": b2, "steered": s2, "improvement": s2 - b2, "sim": float(cos_sim)}

    # 3. Generation-time diff
    print(f"\n[3/4] GENERATION-TIME DIFF...")
    truth_acts, untruth_acts = extract_generation_activations(adapter, test_prompts, base_resps, base_evals, args.layer)
    gen_dir, gen_det = compute_generation_direction(truth_acts, untruth_acts)
    if gen_dir is not None:
        cos_sim_g = np.dot(original_direction, gen_dir)
        print(f"  N truth: {gen_det['n_truthful']}, N untruth: {gen_det['n_untruthful']}, Sim: {cos_sim_g:.3f}")
        b3, s3, _, _, _, _ = evaluate_direction(gen_dir, args.strength)
        print(f"  Base: {b3}/{len(test_ids)}, Steered: {s3}/{len(test_ids)}, Imp: {s3 - b3}")
        results["methods"]["generation_diff"] = {"base": b3, "steered": s3, "improvement": s3 - b3, "sim": float(cos_sim_g)}
    else:
        print(f"  Skipped: {gen_det.get('error')}")

    # 4. Direction search
    if not args.skip_direction_search:
        print(f"\n[4/4] DIRECTION SEARCH...")
        candidates = generate_candidate_directions(pos_np, neg_np, n_random=args.n_random_directions)
        best_imp, best_name = steered_t - base_t, "baseline"
        dir_results = []
        for name, direction in candidates[:8]:
            _, s_count, _, _, _, _ = evaluate_direction(direction, args.strength)
            imp = s_count - base_t
            dir_results.append({"name": name, "steered": s_count, "imp": imp})
            if imp > best_imp:
                best_imp, best_name = imp, name
            print(f"    {name}: {s_count}/{len(test_ids)} (imp: {imp:+d})")
        results["methods"]["direction_search"] = {"best": best_name, "best_imp": best_imp, "all": dir_results}

    # Summary
    print(f"\n{'='*60}")
    best = max(results["methods"].items(), key=lambda x: x[1].get("improvement", -999))
    print(f"BEST METHOD: {best[0]} (improvement: {best[1].get('improvement', 'N/A')})")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output}")
    return results


def _extract_response(raw):
    if "assistant\n\n" in raw:
        return raw.split("assistant\n\n", 1)[-1].strip()
    elif "assistant\n" in raw:
        return raw.split("assistant\n", 1)[-1].strip()
    return raw
