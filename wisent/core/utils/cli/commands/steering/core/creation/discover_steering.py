"""CLI command for steering direction discovery."""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import random
import numpy as np
from pathlib import Path
from wisent.core.utils.config_tools.constants import (
    DISPLAY_TOP_N_BRIEF,
    NORM_EPS,
    SEPARATOR_WIDTH_STANDARD,
    DEFAULT_RANDOM_SEED,
    JSON_INDENT,
)
from wisent.core.primitives.models.config import get_generate_kwargs


def execute_discover_steering(args):
    """Execute the discover-steering command - find optimal steering directions."""
    import torch
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        load_activations_from_hf, load_pair_texts_from_hf,
    )
    from wisent.core.reading.modules.modules.steering.analysis.steering_discovery import (
        discover_behavioral_direction, generate_candidate_directions,
        extract_generation_activations, compute_generation_direction, compare_directions,
    )
    from wisent.core.primitives.model_interface.core.wisent import Wisent
    from wisent.core.reading.evaluators.rotator import EvaluatorRotator
    from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
    from wisent.core.primitives.model_interface.adapters.base import SteeringConfig

    print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
    print("STEERING DIRECTION DISCOVERY")
    print(f"{'='*SEPARATOR_WIDTH_STANDARD}")

    # Load data
    print(f"\nLoading data...")
    all_pair_texts = load_pair_texts_from_hf(
        task_name=args.task, limit=args.discover_task_limit,
    )
    all_pair_ids = list(all_pair_texts.keys())
    random.seed(DEFAULT_RANDOM_SEED)
    random.shuffle(all_pair_ids)

    split_idx = int(len(all_pair_ids) * 0.8)
    train_ids = set(all_pair_ids[:split_idx])
    test_ids = all_pair_ids[split_idx:][:args.n_test_samples]

    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Load reference activations
    pos_ref, neg_ref = load_activations_from_hf(
        model_name=args.model, task_name=args.task, layer=args.layer,
        extraction_strategy=args.extraction_strategy,
        limit=args.discover_train_limit, pair_ids=train_ids,
    )
    pos_np = pos_ref.cpu().numpy()
    neg_np = neg_ref.cpu().numpy()
    print(f"  Loaded {len(pos_ref)} reference pairs")

    # Original steering vector
    original_direction = pos_np.mean(axis=0) - neg_np.mean(axis=0)
    original_direction = original_direction / (np.linalg.norm(original_direction) + NORM_EPS)

    # Load model and evaluator
    print(f"\nLoading model: {args.model}")
    wisent = Wisent.for_text(args.model)
    adapter = wisent.adapter

    EvaluatorRotator.discover_evaluators("wisent.core.reading.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.reading.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task).current
    print(f"  Using evaluator: {evaluator.name}")

    # Get test prompts and references
    test_prompts = [all_pair_texts[pid].get("prompt", "") for pid in test_ids]
    test_pos_refs = [all_pair_texts[pid].get("positive", "") for pid in test_ids]
    test_neg_refs = [all_pair_texts[pid].get("negative", "") for pid in test_ids]

    layer_name = f"layer.{args.layer}"
    results = {"model": args.model, "task": args.task, "layer": args.layer, "methods": {}}

    def evaluate_direction(direction, strength):
        steering_vec = torch.from_numpy(direction).float() * strength
        steering_vectors = LayerActivations({layer_name: steering_vec})
        base_evals, steered_evals, base_resps, steered_resps = [], [], [], []
        for prompt, pos_ref, neg_ref in zip(test_prompts, test_pos_refs, test_neg_refs):
            msgs = [{"role": "user", "content": prompt}]
            fmt = adapter.apply_chat_template(msgs, add_generation_prompt=True)
            gen_kwargs = get_generate_kwargs()
            base_r = _extract_response(adapter._generate_unsteered(fmt, **gen_kwargs))
            base_evals.append(evaluator.evaluate(base_r, pos_ref, correct_answers=[pos_ref], incorrect_answers=[neg_ref]).ground_truth)
            base_resps.append(base_r)
            steer_r = _extract_response(adapter.forward_with_steering(fmt, steering_vectors=steering_vectors, config=SteeringConfig(scale=strength)))
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
        candidates = generate_candidate_directions(pos_np, neg_np, steering_n_random=args.n_random_directions, steering_n_pca=args.steering_n_pca, concept_pca_components=args.concept_pca_components)
        best_imp, best_name = steered_t - base_t, "baseline"
        dir_results = []
        for name, direction in candidates[:DISPLAY_TOP_N_BRIEF]:
            _, s_count, _, _, _, _ = evaluate_direction(direction, args.strength)
            imp = s_count - base_t
            dir_results.append({"name": name, "steered": s_count, "imp": imp})
            if imp > best_imp:
                best_imp, best_name = imp, name
            print(f"    {name}: {s_count}/{len(test_ids)} (imp: {imp:+d})")
        results["methods"]["direction_search"] = {"best": best_name, "best_imp": best_imp, "all": dir_results}

    # Summary
    print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
    best = max(results["methods"].items(), key=lambda x: x[1].get("improvement", -999))
    print(f"BEST METHOD: {best[0]} (improvement: {best[1].get('improvement', 'N/A')})")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=JSON_INDENT)
    print(f"Results saved to: {args.output}")
    return results


def _extract_response(raw):
    if "assistant\n\n" in raw:
        return raw.split("assistant\n\n", 1)[-1].strip()
    elif "assistant\n" in raw:
        return raw.split("assistant\n", 1)[-1].strip()
    return raw
