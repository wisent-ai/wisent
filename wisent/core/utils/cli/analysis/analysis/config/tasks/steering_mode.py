"""Steering mode execution logic for tasks command."""

import sys
import torch

from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.config_tools.constants import JSON_INDENT, NORM_EPS, DISPLAY_TRUNCATION_COMPACT, PROGRESS_LOG_INTERVAL_10, SEPARATOR_WIDTH_STANDARD
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.control.steering_methods.configs.optimal import get_optimal


def execute_steering_mode(args, model, train_pair_set, test_pair_set, collector, extraction_strategy, min_norm_threshold: float, min_clusters: int = None, *, geometry_cv_folds: int, probe_small_hidden: int, probe_mlp_hidden: int, probe_mlp_alpha: float, spectral_n_neighbors: int, direction_n_bootstrap: int, direction_subset_fraction: float, direction_std_penalty: float, consistency_w_cosine: float, consistency_w_positive: float, consistency_w_high_sim: float, sparsity_threshold_fraction: float, detection_threshold: float, direction_moderate_similarity: float):
    """Execute steering mode - train and evaluate steering vectors."""
    from wisent.core.reading.evaluators.rotator import EvaluatorRotator

    print(f"\n🎯 Starting steering evaluation on task: {args.task_names}")
    print(f"   Steering method: {getattr(args, 'steering_method', 'CAA')}")
    print(f"   Steering strength: {getattr(args, 'steering_strength', None)}")

    layer = int(args.layer) if isinstance(args.layer, str) else args.layer
    layer_str = str(layer)
    steering_vector = None

    if hasattr(args, 'load_steering_vector') and args.load_steering_vector:
        steering_vector = _load_steering_vector(args, layer, layer_str)

    task_name = args.task_names[0] if isinstance(args.task_names, list) else args.task_names

    _geo_kw = dict(probe_small_hidden=probe_small_hidden, probe_mlp_hidden=probe_mlp_hidden,
        probe_mlp_alpha=probe_mlp_alpha, spectral_n_neighbors=spectral_n_neighbors,
        direction_n_bootstrap=direction_n_bootstrap, direction_subset_fraction=direction_subset_fraction,
        direction_std_penalty=direction_std_penalty, consistency_w_cosine=consistency_w_cosine,
        consistency_w_positive=consistency_w_positive, consistency_w_high_sim=consistency_w_high_sim,
        sparsity_threshold_fraction=sparsity_threshold_fraction, detection_threshold=detection_threshold,
        direction_moderate_similarity=direction_moderate_similarity)

    if steering_vector is None:
        steering_vector = _compute_steering_vector(
            args, model, train_pair_set, collector, extraction_strategy, layer, layer_str,
            min_clusters=min_clusters,
            geometry_cv_folds=args.geometry_cv_folds,
            **_geo_kw,
        )

    print(f"\n🔧 Initializing evaluator for task '{task_name}'...")
    EvaluatorRotator.discover_evaluators("wisent.core.reading.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.reading.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=task_name, autoload=False)

    return _evaluate_steering(args, model, test_pair_set, steering_vector, layer, layer_str, evaluator, task_name, min_norm_threshold=min_norm_threshold)


def _load_steering_vector(args, layer, layer_str):
    """Load steering vector from file."""
    import json as json_mod

    print(f"\n📂 Loading steering vector from: {args.load_steering_vector}")

    if args.load_steering_vector.endswith('.json'):
        with open(args.load_steering_vector, 'r') as f:
            vector_data = json_mod.load(f)
        steering_vectors = vector_data.get('steering_vectors', {})
        if layer_str in steering_vectors:
            return torch.tensor(steering_vectors[layer_str])
        print(f"   ❌ Layer {layer} not found in vector file")
        sys.exit(1)
    else:
        vector_data = torch.load(args.load_steering_vector)
        steering_vector = vector_data.get('steering_vector', vector_data.get('vector'))
        print(f"   ✓ Loaded steering vector, dim={steering_vector.shape[0]}")
        return steering_vector


def _compute_steering_vector(args, model, train_pair_set, collector, extraction_strategy, layer, layer_str, min_clusters: int = None, *, geometry_cv_folds: int, **geometry_kw):
    """Compute steering vector from training data using zwiad."""
    from wisent.core.reading.modules import compute_geometry_metrics, compute_concept_coherence

    user_method = getattr(args, 'steering_method', None)
    if not user_method:
        raise MissingParameterError(params=["steering_method"], context="--steering-method is required (no auto-selection)")
    recommended_method = user_method.upper()

    print(f"\n🧠 Collecting activations from layer {layer}...")
    positive_activations, negative_activations = [], []

    for i, pair in enumerate(train_pair_set.pairs):
        if i % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"   Processing pair {i+1}/{len(train_pair_set.pairs)}...", end='\r')

        updated_pair = collector.collect(pair, strategy=extraction_strategy, layers=[layer_str])

        if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
            act = updated_pair.positive_response.layers_activations[layer_str]
            if act is not None:
                positive_activations.append(act.cpu().float())

        if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
            act = updated_pair.negative_response.layers_activations[layer_str]
            if act is not None:
                negative_activations.append(act.cpu().float())

    print(f"\n   ✓ Collected {len(positive_activations)} positive and {len(negative_activations)} negative activations")

    pos_tensor = torch.stack(positive_activations)
    neg_tensor = torch.stack(negative_activations)

    print(f"\n🔍 Running zwiad geometry analysis...")
    metrics = compute_geometry_metrics(pos_tensor, neg_tensor, min_clusters=min_clusters, n_folds=geometry_cv_folds, **geometry_kw)
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    print(f"   ├─ Linear probe accuracy: {metrics['linear_probe_accuracy']:.3f}")
    print(f"   ├─ Concept coherence:     {coherence:.3f}")
    print(f"   └─ Method:                {recommended_method}")

    return _train_steering_method(args, model, recommended_method, pos_tensor, neg_tensor, layer, collector, extraction_strategy, train_pair_set)


def _train_steering_method(args, model, method, pos_tensor, neg_tensor, layer, collector, extraction_strategy, train_pair_set):
    """Train steering vector using the specified method."""
    print(f"\n🎯 Training steering using {method}...")

    if method == "CAA":
        pos_mean, neg_mean = pos_tensor.mean(dim=0), neg_tensor.mean(dim=0)
        steering_vector = pos_mean - neg_mean
        if getattr(args, 'caa_normalize', get_optimal("normalize")):
            steering_vector = steering_vector / (steering_vector.norm() + NORM_EPS)
        print(f"   ✓ CAA steering vector computed, norm={steering_vector.norm().item():.4f}")
        return steering_vector

    elif method == "GROM":
        return _train_grom(args, model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor, args.enrichment_max_pairs)

    elif method == "TECZA":
        return _train_tecza(args, model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor, args.enrichment_max_pairs)

    else:
        print(f"   ⚠️  Unknown method {method}, using CAA")
        steering_vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        steering_vector = steering_vector / (steering_vector.norm() + NORM_EPS)
        return steering_vector


def _train_grom(args, model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor, enrichment_max_pairs: int):
    """Train GROM steering vector."""
    from wisent.core.control.steering_methods.methods.grom import GROMMethod
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet

    all_layers = [str(i) for i in range(1, model.num_layers + 1)]
    enriched_pairs = [collector.collect(pair, strategy=extraction_strategy, layers=all_layers) for pair in train_pair_set.pairs[:enrichment_max_pairs]]
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="grom_training")

    grom_num_directions = getattr(args, 'grom_num_directions', None)
    if grom_num_directions is None:
        raise MissingParameterError(params=["grom_num_directions"], context="GROM training requires --grom-num-directions")
    grom_method = GROMMethod(model=model, num_directions=grom_num_directions, manifold_method="pca",
                               steering_layers=[int(l) for l in all_layers], sensor_layer=1)
    grom_result = grom_method.train_grom(pair_set)
    layer_key = f"layer_{layer}"

    if layer_key in grom_result.directions:
        dirs, weights = grom_result.directions[layer_key], grom_result.direction_weights[layer_key]
        steering_vector = (dirs * (weights / (weights.sum() + NORM_EPS)).unsqueeze(-1)).sum(dim=0)
    else:
        steering_vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)

    steering_vector = steering_vector / (steering_vector.norm() + NORM_EPS)
    print(f"   ✓ GROM steering vector computed, norm={steering_vector.norm().item():.4f}")
    return steering_vector


def _train_tecza(args, model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor, enrichment_max_pairs: int):
    """Train TECZA steering vector."""
    from wisent.core.control.steering_methods.methods.advanced import TECZAMethod
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet

    all_layers = [str(i) for i in range(1, model.num_layers + 1)]
    enriched_pairs = [collector.collect(pair, strategy=extraction_strategy, layers=all_layers) for pair in train_pair_set.pairs[:enrichment_max_pairs]]
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="tecza_training")

    def _rq(name):
        val = getattr(args, name, None)
        if val is None:
            raise MissingParameterError(params=[name], context=f"TECZA training requires --{name.replace('_', '-')}")
        return val
    tecza_method = TECZAMethod(
        model=model.hf_model, num_directions=_rq('tecza_num_directions'),
        optimization_steps=_rq('tecza_optimization_steps'), learning_rate=_rq('tecza_learning_rate'),
        retain_weight=_rq('tecza_retain_weight'), independence_weight=_rq('tecza_independence_weight'),
        min_cosine_similarity=_rq('tecza_min_cosine_sim'), max_cosine_similarity=_rq('tecza_max_cosine_sim'),
        variance_threshold=_rq('tecza_variance_threshold'), marginal_threshold=_rq('tecza_marginal_threshold'),
        max_directions=_rq('tecza_max_directions'), ablation_weight=_rq('tecza_ablation_weight'),
        addition_weight=_rq('tecza_addition_weight'), separation_margin=_rq('tecza_separation_margin'),
        perturbation_scale=_rq('tecza_perturbation_scale'), universal_basis_noise=_rq('tecza_universal_basis_noise'),
        log_interval=_rq('tecza_log_interval'),
    )
    tecza_result = tecza_method.train(pair_set)
    layer_key = f"layer_{layer}"

    if layer_key in tecza_result.directions:
        steering_vector = tecza_result.directions[layer_key][0]
    else:
        steering_vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)

    steering_vector = steering_vector / (steering_vector.norm() + NORM_EPS)
    print(f"   ✓ TECZA steering vector computed, norm={steering_vector.norm().item():.4f}")
    return steering_vector


def _evaluate_steering(args, model, test_pair_set, steering_vector, layer, layer_str, evaluator, task_name, min_norm_threshold):
    """Evaluate steering effectiveness on test set."""
    import os
    import json

    print(f"\n📊 Evaluating on {len(test_pair_set.pairs)} test pairs...")
    baseline_correct, steered_correct, total = 0, 0, 0
    results = []
    steering_strength = getattr(args, 'steering_strength', None)

    for i, pair in enumerate(test_pair_set.pairs):
        print(f"   Processing {i+1}/{len(test_pair_set.pairs)}...", end='\r')

        question, expected = pair.prompt, pair.positive_response.model_response
        choices = [pair.negative_response.model_response, pair.positive_response.model_response]
        messages = [{"role": "user", "content": question}]

        resp_base = model.generate([messages], **get_generate_kwargs())[0]
        eval_kwargs = {'response': resp_base, 'expected': expected, 'model': model, 'question': question, 'choices': choices, 'task_name': task_name}
        if hasattr(pair, 'metadata') and pair.metadata:
            eval_kwargs.update({k: v for k, v in pair.metadata.items() if v is not None and k not in eval_kwargs})
        base_correct = evaluator.evaluate(**eval_kwargs).ground_truth == "TRUTHFUL"

        model.set_steering_from_raw({layer_str: steering_vector}, scale=steering_strength, min_norm_threshold=min_norm_threshold, normalize=False)
        resp_steer = model.generate([messages], **get_generate_kwargs())[0]
        model.clear_steering()

        eval_kwargs['response'] = resp_steer
        steer_correct = evaluator.evaluate(**eval_kwargs).ground_truth == "TRUTHFUL"

        baseline_correct += int(base_correct)
        steered_correct += int(steer_correct)
        results.append({'question': question[:DISPLAY_TRUNCATION_COMPACT], 'baseline_correct': base_correct, 'steered_correct': steer_correct})
        total += 1

    print(f"\n\n{'='*SEPARATOR_WIDTH_STANDARD}\n📊 STEERING EVALUATION RESULTS\n{'='*SEPARATOR_WIDTH_STANDARD}")
    print(f"   Baseline accuracy:  {baseline_correct}/{total} ({100*baseline_correct/total:.1f}%)")
    print(f"   Steered accuracy:   {steered_correct}/{total} ({100*steered_correct/total:.1f}%)")
    print(f"   Delta:              {steered_correct - baseline_correct:+d} ({100*(steered_correct-baseline_correct)/total:+.1f}%)")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'steering_evaluation.json'), 'w') as f:
            json.dump({'task': task_name, 'layer': layer, 'baseline_accuracy': baseline_correct/total,
                      'steered_accuracy': steered_correct/total, 'delta': (steered_correct-baseline_correct)/total}, f, indent=JSON_INDENT)

    return {'task': task_name, 'baseline_accuracy': baseline_correct/total, 'steered_accuracy': steered_correct/total}
