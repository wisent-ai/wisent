"""Classifier training and evaluation logic for tasks command."""

import os
import json
import numpy as np
import torch

from wisent.core.models import get_generate_kwargs
from wisent.core.errors import UnknownTypeError


def collect_activations(args, model, pair_set, ActivationCollector, ExtractionStrategy):
    """Collect activations from all pairs."""
    layer = int(args.layer) if isinstance(args.layer, str) else args.layer
    layer_str = str(layer)

    print(f"\n🧠 Extracting activations from layer {layer}...")
    collector = ActivationCollector(model=model)
    extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', 'chat_last'))
    print(f"   Extraction strategy: {extraction_strategy.value}")

    positive_activations, negative_activations = [], []

    for i, pair in enumerate(pair_set.pairs):
        if i % 10 == 0:
            print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...", end='\r')

        updated_pair = collector.collect(pair, strategy=extraction_strategy, layers=[layer_str])

        if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
            act = updated_pair.positive_response.layers_activations[layer_str]
            if act is not None:
                positive_activations.append(act.cpu().float().numpy())

        if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
            act = updated_pair.negative_response.layers_activations[layer_str]
            if act is not None:
                negative_activations.append(act.cpu().float().numpy())

    print(f"\n   ✓ Collected {len(positive_activations)} positive and {len(negative_activations)} negative activations")

    return {
        'positive': positive_activations,
        'negative': negative_activations,
        'layer': layer,
        'layer_str': layer_str,
        'collector': collector,
        'extraction_strategy': extraction_strategy
    }


def train_steering_vector(args, activations):
    """Train and save steering vector."""
    from wisent.core.steering_methods.methods.caa import CAAMethod

    print(f"\n🎯 Training steering vector using {args.steering_method} method...")

    pos_tensors = [torch.from_numpy(act).float() for act in activations['positive']]
    neg_tensors = [torch.from_numpy(act).float() for act in activations['negative']]

    steering_method = CAAMethod(normalize=True)
    steering_vector = steering_method.train_for_layer(pos_tensors, neg_tensors)

    print(f"\n💾 Saving steering vector to '{args.save_steering_vector}'...")
    os.makedirs(os.path.dirname(args.save_steering_vector) or '.', exist_ok=True)
    torch.save({
        'steering_vector': steering_vector, 'layer_index': activations['layer'],
        'method': args.steering_method, 'model': args.model, 'task': args.task_names,
        'vector': steering_vector, 'layer': activations['layer'],
    }, args.save_steering_vector)
    print(f"   ✓ Steering vector saved to: {args.save_steering_vector}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'training_report.json'), 'w') as f:
            json.dump({'method': args.steering_method, 'layer': activations['layer'],
                      'num_positive': len(activations['positive']), 'num_negative': len(activations['negative']),
                      'vector_shape': list(steering_vector.shape)}, f, indent=2)

    return {
        "steering_vector_saved": True, "vector_path": args.save_steering_vector,
        "layer": activations['layer'], "method": args.steering_method,
        "num_positive": len(activations['positive']), "num_negative": len(activations['negative']),
        "vector_shape": list(steering_vector.shape), "test_accuracy": None, "test_f1_score": None,
    }


def train_classifier(args, activations, LogisticClassifier, MLPClassifier, ClassifierTrainConfig):
    """Train the classifier on collected activations."""
    print(f"\n🎯 Preparing training data...")
    X_positive = np.array(activations['positive'])
    X_negative = np.array(activations['negative'])
    X = np.vstack([X_positive, X_negative])
    y = np.array([1] * len(activations['positive']) + [0] * len(activations['negative']))

    print(f"   Training set: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Positive samples: {sum(y == 1)}, Negative samples: {sum(y == 0)}")

    print(f"\n🏋️  Training {args.classifier_type} classifier...")
    if args.classifier_type == 'logistic':
        classifier = LogisticClassifier(threshold=args.detection_threshold, device=args.device)
    elif args.classifier_type == 'mlp':
        classifier = MLPClassifier(threshold=args.detection_threshold, device=args.device)
    else:
        raise UnknownTypeError(entity_type="classifier_type", value=args.classifier_type, valid_values=["logistic", "mlp"])

    train_config = ClassifierTrainConfig(
        test_size=1.0 - args.split_ratio, num_epochs=50, batch_size=32,
        learning_rate=1e-3, monitor='f1', random_state=args.seed
    )

    report = classifier.fit(X, y, config=train_config)
    print(f"\n📈 Training completed! Best epoch: {report.best_epoch}/{report.epochs_ran}")

    return classifier, report


def evaluate_classifier(args, model, classifier, test_pairs, activations, task_name, eval_task_name, DetectionHandler, DetectionAction, evaluate_quality):
    """Evaluate classifier on real model generations."""
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair

    print(f"\n🎯 Evaluating classifier on real model generations...")
    eval_task_for_evaluator = eval_task_name if eval_task_name else task_name

    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=eval_task_for_evaluator, autoload=False)

    detection_handler = _setup_detection_handler(args, DetectionHandler, DetectionAction)
    detection_stats = {'total_outputs': 0, 'issues_detected': 0, 'low_quality_outputs': 0, 'handled_outputs': 0, 'detection_types': {}}
    enable_quality_check = hasattr(args, 'enable_quality_check') and args.enable_quality_check
    quality_threshold = getattr(args, 'quality_threshold', 50.0)

    gen_collector = ActivationCollector(model=model)
    generation_results = []

    for i, pair in enumerate(test_pairs.pairs):
        if i % 10 == 0:
            print(f"      Processing {i+1}/{len(test_pairs.pairs)}...", end='\r')

        result = _evaluate_single_generation(
            pair, model, classifier, evaluator, gen_collector, activations,
            detection_handler, detection_stats, enable_quality_check, quality_threshold,
            evaluate_quality, task_name, args
        )
        if result:
            generation_results.append(result)

    print(f"\n   ✓ Evaluated {len(generation_results)} generations")
    return generation_results, detection_stats


def _setup_detection_handler(args, DetectionHandler, DetectionAction):
    """Setup detection handler if enabled."""
    if not hasattr(args, 'detection_action') or args.detection_action == 'pass_through':
        return None

    action_map = {
        'pass_through': DetectionAction.PASS_THROUGH,
        'replace_with_placeholder': DetectionAction.REPLACE_WITH_PLACEHOLDER,
        'regenerate_until_safe': DetectionAction.REGENERATE_UNTIL_SAFE
    }

    return DetectionHandler(
        action=action_map.get(args.detection_action, DetectionAction.REPLACE_WITH_PLACEHOLDER),
        placeholder_message=getattr(args, 'placeholder_message', None),
        max_regeneration_attempts=getattr(args, 'max_regeneration_attempts', 3),
        log_detections=True
    )


def _evaluate_single_generation(pair, model, classifier, evaluator, gen_collector, activations, detection_handler, detection_stats, enable_quality_check, quality_threshold, evaluate_quality, task_name, args):
    """Evaluate a single generation."""
    from wisent.core.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair

    question = pair.prompt
    expected = pair.positive_response.model_response
    choices = [pair.negative_response.model_response, pair.positive_response.model_response]

    messages = [{"role": "user", "content": question}]
    response = model.generate([messages], **get_generate_kwargs())[0]

    eval_kwargs = {'response': response, 'expected': expected, 'model': model, 'question': question, 'choices': choices, 'task_name': task_name}
    if hasattr(pair, 'metadata') and pair.metadata:
        eval_kwargs.update({k: v for k, v in pair.metadata.items() if v is not None and k not in eval_kwargs})
    eval_result = evaluator.evaluate(**eval_kwargs)

    temp_pair = ContrastivePair(
        prompt=question,
        positive_response=PositiveResponse(model_response=response, layers_activations={}),
        negative_response=NegativeResponse(model_response="placeholder", layers_activations={}),
        label=None, trait_description=None
    )

    collected_full = gen_collector.collect(temp_pair, strategy=activations['extraction_strategy'])

    if not collected_full.positive_response.layers_activations:
        return None

    layer_activations = collected_full.positive_response.layers_activations
    if activations['layer_str'] not in layer_activations:
        return None

    activation = layer_activations[activations['layer_str']]
    if activation is None or not isinstance(activation, torch.Tensor):
        return None

    act_tensor = activation.unsqueeze(0).float()
    pred_proba_result = classifier.predict_proba(act_tensor)
    pred_proba = pred_proba_result if isinstance(pred_proba_result, float) else pred_proba_result[0]
    pred_label = int(pred_proba > args.detection_threshold)

    detection_stats['total_outputs'] += 1
    ground_truth = 1 if eval_result.ground_truth == "TRUTHFUL" else 0

    return {
        'question': question, 'response': response, 'expected': expected,
        'eval_result': eval_result.ground_truth, 'classifier_pred': pred_label,
        'classifier_proba': float(pred_proba), 'correct': pred_label == ground_truth,
    }


def compute_metrics(generation_results):
    """Compute evaluation metrics from generation results."""
    if not generation_results:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    correct = sum(1 for r in generation_results if r['correct'])
    accuracy = correct / len(generation_results)

    tp = sum(1 for r in generation_results if r['classifier_pred'] == 1 and r['eval_result'] == 'TRUTHFUL')
    fp = sum(1 for r in generation_results if r['classifier_pred'] == 1 and r['eval_result'] == 'UNTRUTHFUL')
    fn = sum(1 for r in generation_results if r['classifier_pred'] == 0 and r['eval_result'] == 'TRUTHFUL')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}


def save_classifier_and_results(args, classifier, report, activations, generation_results, detection_stats, metrics, ModelPersistence, create_classifier_metadata):
    """Save classifier and output artifacts."""
    if args.save_classifier:
        print(f"\n💾 Saving classifier to '{args.save_classifier}'...")
        metadata = create_classifier_metadata(
            model_name=args.model, task_name=args.task_names, layer=activations['layer'],
            classifier_type=args.classifier_type, training_accuracy=report.final.accuracy,
            training_samples=len(activations['positive']) + len(activations['negative']),
            token_aggregation=activations['extraction_strategy'].value, detection_threshold=args.detection_threshold
        )
        save_path = ModelPersistence.save_classifier(classifier=classifier, layer=activations['layer'], save_path=args.save_classifier, metadata=metadata)
        print(f"   ✓ Classifier saved to: {save_path}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'training_report.json'), 'w') as f:
            json.dump(report.asdict(), f, indent=2)

        if generation_results:
            with open(os.path.join(args.output, 'generation_details.json'), 'w') as f:
                json.dump({
                    'task': args.task_names, 'model': args.model, 'layer': activations['layer'],
                    'aggregation': activations['extraction_strategy'].value, 'threshold': args.detection_threshold,
                    'num_generations': len(generation_results), 'detection_stats': detection_stats,
                    'generations': generation_results
                }, f, indent=2)

    print(f"\n   📊 Real-world performance:")
    print(f"     • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     • F1 Score:  {metrics['f1']:.4f}")
    print(f"\n✅ Task completed successfully!\n")

    return {
        "accuracy": float(metrics['accuracy']), "f1_score": float(metrics['f1']),
        "precision": float(metrics['precision']), "recall": float(metrics['recall']),
        "generation_count": len(generation_results), "best_epoch": report.best_epoch,
        "epochs_ran": report.epochs_ran, "generation_details": generation_results
    }
