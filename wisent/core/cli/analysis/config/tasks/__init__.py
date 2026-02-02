"""Tasks command execution logic - refactored into modules."""

import sys

from wisent.core.models import get_generate_kwargs
from wisent.core.errors import UnknownTypeError


def execute_tasks(args):
    """Execute the tasks command - train classifier or steering on benchmark tasks."""
    from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy
    from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
    from wisent.core.classifiers.classifiers.models.mlp import MLPClassifier
    from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig
    from wisent.core.models.model_persistence import ModelPersistence, create_classifier_metadata
    from wisent.core.utils.display.detection_handling import DetectionHandler, DetectionAction
    from wisent.core.evaluators.personalization.coherence import evaluate_quality

    from .utilities import execute_list_tasks, execute_task_info, select_tasks_by_criteria, execute_optimization
    from .cross_benchmark import load_cross_benchmark_data, load_single_task_data
    from .classifier import (
        collect_activations, train_steering_vector, train_classifier,
        evaluate_classifier, compute_metrics, save_classifier_and_results
    )
    from .steering_mode import execute_steering_mode

    # Handle mode selection
    steering_mode = hasattr(args, 'steering_mode') and args.steering_mode
    classification_mode = hasattr(args, 'classification_mode') and args.classification_mode

    special_flags = [
        hasattr(args, 'list_tasks') and args.list_tasks,
        hasattr(args, 'task_info') and args.task_info,
        hasattr(args, 'inference_only') and args.inference_only,
        hasattr(args, 'optimize') and args.optimize,
        hasattr(args, 'cross_benchmark') and args.cross_benchmark,
        hasattr(args, 'train_only') and args.train_only,
    ]

    if not any(special_flags) and not steering_mode and not classification_mode:
        _prompt_for_mode(args)
        steering_mode = hasattr(args, 'steering_mode') and args.steering_mode
        classification_mode = hasattr(args, 'classification_mode') and args.classification_mode

    # Handle inference-only mode
    if args.inference_only and args.load_steering_vector:
        return _handle_inference_only(args)

    # Handle --list-tasks flag
    if hasattr(args, 'list_tasks') and args.list_tasks:
        return execute_list_tasks(args, LMEvalDataLoader)

    # Handle --task-info flag
    if hasattr(args, 'task_info') and args.task_info:
        return execute_task_info(args, LMEvalDataLoader)

    # Handle --skills or --risks task selection
    if (hasattr(args, 'skills') and args.skills) or (hasattr(args, 'risks') and args.risks):
        select_tasks_by_criteria(args)

    # Load model
    print(f"\n🤖 Loading model '{args.model}'...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Model loaded with {model.num_layers} layers")

    # Handle optimization mode
    if hasattr(args, 'optimize') and args.optimize:
        return execute_optimization(args, model, LMEvalDataLoader)

    # Load task data
    if hasattr(args, 'cross_benchmark') and args.cross_benchmark:
        if not (hasattr(args, 'train_task') and args.train_task and hasattr(args, 'eval_task') and args.eval_task):
            print(f"❌ Error: --cross-benchmark requires both --train-task and --eval-task")
            sys.exit(1)
        data = load_cross_benchmark_data(args, LMEvalDataLoader)
    else:
        data = load_single_task_data(args, LMEvalDataLoader)

    train_pair_set = data['train_pair_set']
    test_pair_set = data['test_pair_set']
    task_name = data['task_name']
    eval_task_name = data['eval_task_name']

    # Handle steering mode
    if steering_mode:
        collector = ActivationCollector(model=model)
        extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', 'chat_last'))
        return execute_steering_mode(args, model, train_pair_set, test_pair_set, collector, extraction_strategy)

    # Collect activations
    activations = collect_activations(args, model, train_pair_set, ActivationCollector, ExtractionStrategy)

    # Handle steering vector mode
    if args.save_steering_vector and args.train_only:
        return train_steering_vector(args, activations)

    # Train classifier
    classifier, report = train_classifier(args, activations, LogisticClassifier, MLPClassifier, ClassifierTrainConfig)

    # Evaluate classifier
    generation_results, detection_stats = evaluate_classifier(
        args, model, classifier, test_pair_set, activations, task_name, eval_task_name,
        DetectionHandler, DetectionAction, evaluate_quality
    )

    # Compute metrics and save
    metrics = compute_metrics(generation_results)
    return save_classifier_and_results(
        args, classifier, report, activations, generation_results, detection_stats, metrics,
        ModelPersistence, create_classifier_metadata
    )


def _prompt_for_mode(args):
    """Prompt user to select steering or classification mode."""
    print("\n" + "=" * 60)
    print("MODE SELECTION REQUIRED")
    print("=" * 60)
    print("\nThe 'tasks' command can run in two modes:\n")
    print("  1. STEERING MODE (--steering-mode)")
    print("     Train steering vectors to modify model behavior\n")
    print("  2. CLASSIFICATION MODE (--classification-mode)")
    print("     Train a classifier to detect good/bad responses\n")
    print("=" * 60)

    while True:
        try:
            choice = input("\nSelect mode [s]teering or [c]lassification (s/c): ").strip().lower()
            if choice in ['s', 'steering']:
                args.steering_mode = True
                print("\n→ Running in STEERING mode\n")
                break
            elif choice in ['c', 'classification']:
                args.classification_mode = True
                print("\n→ Running in CLASSIFICATION mode\n")
                break
            else:
                print("  Please enter 's' for steering or 'c' for classification")
        except (EOFError, KeyboardInterrupt):
            print("\n\nAborted. Please specify --steering-mode or --classification-mode")
            sys.exit(1)


def _handle_inference_only(args):
    """Handle inference-only mode with steering vector."""
    import torch

    print(f"\n🎯 Starting inference with steering vector")
    print(f"   Loading vector from: {args.load_steering_vector}")

    vector_data = torch.load(args.load_steering_vector)
    steering_vector = vector_data['vector']
    layer = vector_data['layer']

    print(f"   ✓ Loaded steering vector for layer {layer}")
    print(f"   Model: {vector_data.get('model', 'unknown')}")
    print(f"   Method: {vector_data.get('method', 'unknown')}")
    print(f"\n✅ Steering vector loaded successfully!\n")

    return {
        "steering_vector_loaded": True, "vector_path": args.load_steering_vector,
        "layer": layer, "method": vector_data.get('method', 'unknown'),
        "test_accuracy": None, "test_f1_score": None, "training_time": 0.0, "evaluation_results": {}
    }
