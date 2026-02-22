"""Steering mode execution logic for tasks command."""

import sys
import torch

from wisent.core.models import get_generate_kwargs


def execute_steering_mode(args, model, train_pair_set, test_pair_set, collector, extraction_strategy):
    """Execute steering mode - train and evaluate steering vectors."""
    from wisent.core.evaluators.rotator import EvaluatorRotator

    print(f"\n🎯 Starting steering evaluation on task: {args.task_names}")
    print(f"   Steering method: {getattr(args, 'steering_method', 'CAA')}")
    print(f"   Steering strength: {getattr(args, 'steering_strength', 1.0)}")

    layer = int(args.layer) if isinstance(args.layer, str) else args.layer
    layer_str = str(layer)
    steering_vector = None

    if hasattr(args, 'load_steering_vector') and args.load_steering_vector:
        steering_vector = _load_steering_vector(args, layer, layer_str)

    task_name = args.task_names[0] if isinstance(args.task_names, list) else args.task_names

    if steering_vector is None:
        steering_vector = _compute_steering_vector(
            args, model, train_pair_set, collector, extraction_strategy, layer, layer_str
        )

    print(f"\n🔧 Initializing evaluator for task '{task_name}'...")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=task_name, autoload=False)

    return _evaluate_steering(args, model, test_pair_set, steering_vector, layer, layer_str, evaluator, task_name)


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


def _compute_steering_vector(args, model, train_pair_set, collector, extraction_strategy, layer, layer_str):
    """Compute steering vector from training data using zwiad."""
    from wisent.core.geometry import compute_geometry_metrics, compute_recommendation, compute_concept_coherence

    print(f"\n🧠 Collecting activations from layer {layer}...")
    positive_activations, negative_activations = [], []

    for i, pair in enumerate(train_pair_set.pairs):
        if i % 10 == 0:
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
    metrics = compute_geometry_metrics(pos_tensor, neg_tensor, n_folds=3)
    recommendation = compute_recommendation(metrics)
    recommended_method = recommendation.get("recommended_method", "CAA").upper()
    confidence = recommendation.get("confidence", 0.5)
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    print(f"   ├─ Linear probe accuracy: {metrics.get('linear_probe_accuracy', 0):.3f}")
    print(f"   ├─ Concept coherence:     {coherence:.3f}")
    print(f"   └─ Recommendation:        {recommended_method} (confidence={confidence:.2f})")

    user_method = getattr(args, 'steering_method', 'auto')
    if user_method and user_method.lower() != 'auto':
        recommended_method = user_method.upper()
        print(f"   → User override: using {recommended_method}")

    return _train_steering_method(args, model, recommended_method, pos_tensor, neg_tensor, layer, collector, extraction_strategy, train_pair_set)


def _train_steering_method(args, model, method, pos_tensor, neg_tensor, layer, collector, extraction_strategy, train_pair_set):
    """Train steering vector using the specified method."""
    print(f"\n🎯 Training steering using {method}...")

    if method == "CAA":
        pos_mean, neg_mean = pos_tensor.mean(dim=0), neg_tensor.mean(dim=0)
        steering_vector = pos_mean - neg_mean
        if getattr(args, 'caa_normalize', True):
            steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        print(f"   ✓ CAA steering vector computed, norm={steering_vector.norm().item():.4f}")
        return steering_vector

    elif method == "GROM":
        return _train_grom(model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor)

    elif method == "TECZA":
        return _train_tecza(model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor)

    else:
        print(f"   ⚠️  Unknown method {method}, using CAA")
        steering_vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        return steering_vector


def _train_grom(model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor):
    """Train GROM steering vector."""
    from wisent.core.steering_methods.methods.grom import GROMMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    all_layers = [str(i) for i in range(1, model.num_layers + 1)]
    enriched_pairs = [collector.collect(pair, strategy=extraction_strategy, layers=all_layers) for pair in train_pair_set.pairs[:50]]
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="grom_training")

    grom_method = GROMMethod(model=model, num_directions=8, manifold_method="pca",
                               steering_layers=[int(l) for l in all_layers], sensor_layer=1)
    grom_result = grom_method.train_grom(pair_set)
    layer_key = f"layer_{layer}"

    if layer_key in grom_result.directions:
        dirs, weights = grom_result.directions[layer_key], grom_result.direction_weights[layer_key]
        steering_vector = (dirs * (weights / (weights.sum() + 1e-8)).unsqueeze(-1)).sum(dim=0)
    else:
        steering_vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)

    steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
    print(f"   ✓ GROM steering vector computed, norm={steering_vector.norm().item():.4f}")
    return steering_vector


def _train_tecza(model, layer, collector, extraction_strategy, train_pair_set, pos_tensor, neg_tensor):
    """Train TECZA steering vector."""
    from wisent.core.steering_methods.methods.advanced import TECZAMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    all_layers = [str(i) for i in range(1, model.num_layers + 1)]
    enriched_pairs = [collector.collect(pair, strategy=extraction_strategy, layers=all_layers) for pair in train_pair_set.pairs[:50]]
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="tecza_training")

    tecza_method = TECZAMethod(model=model.hf_model, num_directions=3)
    tecza_result = tecza_method.train(pair_set)
    layer_key = f"layer_{layer}"

    if layer_key in tecza_result.directions:
        steering_vector = tecza_result.directions[layer_key][0]
    else:
        steering_vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)

    steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
    print(f"   ✓ TECZA steering vector computed, norm={steering_vector.norm().item():.4f}")
    return steering_vector


def _evaluate_steering(args, model, test_pair_set, steering_vector, layer, layer_str, evaluator, task_name):
    """Evaluate steering effectiveness on test set."""
    import os
    import json

    print(f"\n📊 Evaluating on {len(test_pair_set.pairs)} test pairs...")
    baseline_correct, steered_correct, total = 0, 0, 0
    results = []
    steering_strength = getattr(args, 'steering_strength', 1.0)

    for i, pair in enumerate(test_pair_set.pairs):
        print(f"   Processing {i+1}/{len(test_pair_set.pairs)}...", end='\r')

        question, expected = pair.prompt, pair.positive_response.model_response
        choices = [pair.negative_response.model_response, pair.positive_response.model_response]
        messages = [{"role": "user", "content": question}]

        resp_base = model.generate([messages], **get_generate_kwargs(max_new_tokens=512))[0]
        eval_kwargs = {'response': resp_base, 'expected': expected, 'model': model, 'question': question, 'choices': choices, 'task_name': task_name}
        if hasattr(pair, 'metadata') and pair.metadata:
            eval_kwargs.update({k: v for k, v in pair.metadata.items() if v is not None and k not in eval_kwargs})
        base_correct = evaluator.evaluate(**eval_kwargs).ground_truth == "TRUTHFUL"

        model.set_steering_from_raw({layer_str: steering_vector}, scale=steering_strength, normalize=False)
        resp_steer = model.generate([messages], **get_generate_kwargs(max_new_tokens=512))[0]
        model.clear_steering()

        eval_kwargs['response'] = resp_steer
        steer_correct = evaluator.evaluate(**eval_kwargs).ground_truth == "TRUTHFUL"

        baseline_correct += int(base_correct)
        steered_correct += int(steer_correct)
        results.append({'question': question[:100], 'baseline_correct': base_correct, 'steered_correct': steer_correct})
        total += 1

    print(f"\n\n{'='*60}\n📊 STEERING EVALUATION RESULTS\n{'='*60}")
    print(f"   Baseline accuracy:  {baseline_correct}/{total} ({100*baseline_correct/total:.1f}%)")
    print(f"   Steered accuracy:   {steered_correct}/{total} ({100*steered_correct/total:.1f}%)")
    print(f"   Delta:              {steered_correct - baseline_correct:+d} ({100*(steered_correct-baseline_correct)/total:+.1f}%)")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'steering_evaluation.json'), 'w') as f:
            json.dump({'task': task_name, 'layer': layer, 'baseline_accuracy': baseline_correct/total,
                      'steered_accuracy': steered_correct/total, 'delta': (steered_correct-baseline_correct)/total}, f, indent=2)

    return {'task': task_name, 'baseline_accuracy': baseline_correct/total, 'steered_accuracy': steered_correct/total}
