"""Tasks command execution logic."""

import sys
import os
import json
import numpy as np

from wisent.core.models.inference_config import get_generate_kwargs
from wisent.core.errors import UnknownTypeError


def execute_tasks(args):
    """Execute the tasks command - train classifier on benchmark tasks."""
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
    from wisent.core.classifiers.classifiers.models.mlp import MLPClassifier
    from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig
    from wisent.core.model_persistence import ModelPersistence, create_classifier_metadata
    from wisent.core.detection_handling import DetectionHandler, DetectionAction
    from wisent.core.evaluators.personalization.coherence import evaluate_quality

    # Check if this is inference-only mode with steering vector
    if args.inference_only and args.load_steering_vector:
        import torch
        print(f"\n🎯 Starting inference with steering vector")
        print(f"   Loading vector from: {args.load_steering_vector}")

        # Load steering vector
        vector_data = torch.load(args.load_steering_vector)
        steering_vector = vector_data['vector']
        layer = vector_data['layer']

        print(f"   ✓ Loaded steering vector for layer {layer}")
        print(f"   Model: {vector_data.get('model', 'unknown')}")
        print(f"   Method: {vector_data.get('method', 'unknown')}")

        # For now, just load and validate - actual inference would require more implementation
        print(f"\n✅ Steering vector loaded successfully!\n")
        print(f"Note: Inference with steering vector requires additional implementation")

        # Return results for programmatic access
        return {
            "steering_vector_loaded": True,
            "vector_path": args.load_steering_vector,
            "layer": layer,
            "method": vector_data.get('method', 'unknown'),
            "test_accuracy": None,
            "test_f1_score": None,
            "training_time": 0.0,
            "evaluation_results": {}
        }

    # Handle --list-tasks flag
    if hasattr(args, 'list_tasks') and args.list_tasks:
        from wisent.core.task_selector import TaskSelector
        selector = TaskSelector()

        # Get tasks based on skills/risks filters
        if hasattr(args, 'skills') and args.skills:
            print(f"\n📋 Tasks matching skills: {', '.join(args.skills)}")
            tasks = selector.find_tasks_by_tags(
                skills=args.skills,
                min_quality_score=getattr(args, 'min_quality_score', 2)
            )
        elif hasattr(args, 'risks') and args.risks:
            print(f"\n📋 Tasks matching risks: {', '.join(args.risks)}")
            tasks = selector.find_tasks_by_tags(
                risks=args.risks,
                min_quality_score=getattr(args, 'min_quality_score', 2)
            )
        else:
            print(f"\n📋 All available tasks:")
            tasks = selector.find_tasks_by_tags(min_quality_score=getattr(args, 'min_quality_score', 2))

        print(f"\n   Found {len(tasks)} tasks:\n")
        for i, task in enumerate(sorted(tasks), 1):
            print(f"   {i}. {task}")
        print()
        return {"tasks": tasks}

    # Handle --task-info flag
    if hasattr(args, 'task_info') and args.task_info:
        task_name = args.task_info
        print(f"\n📋 Task Information: {task_name}")
        print(f"{'='*60}")

        try:
            loader = LMEvalDataLoader()
            task_obj = loader.load_lm_eval_task(task_name)

            if isinstance(task_obj, dict):
                # Group task
                print(f"\n   Type: Group Task")
                print(f"   Subtasks: {len(task_obj)}")
                for subname in task_obj.keys():
                    print(f"      • {subname}")
            else:
                # Single task
                print(f"\n   Type: Single Task")
                if hasattr(task_obj, 'config'):
                    config = task_obj.config
                    if hasattr(config, 'description') and config.description:
                        print(f"   Description: {config.description}")
                    if hasattr(config, 'dataset_name') and config.dataset_name:
                        print(f"   Dataset: {config.dataset_name}")
                    if hasattr(config, 'metric_list') and config.metric_list:
                        metrics = [m.get('metric', str(m)) if isinstance(m, dict) else str(m) for m in config.metric_list]
                        print(f"   Metrics: {', '.join(metrics)}")
                    if hasattr(config, 'num_fewshot'):
                        print(f"   Few-shot examples: {config.num_fewshot}")

            print()
            return {"task": task_name, "found": True}

        except Exception as e:
            print(f"\n   ❌ Error loading task: {e}")
            print()
            return {"task": task_name, "found": False, "error": str(e)}

    # Handle --skills or --risks task selection
    if (hasattr(args, 'skills') and args.skills) or (hasattr(args, 'risks') and args.risks):
        from wisent.core.task_selector import TaskSelector
        selector = TaskSelector()

        selected_tasks = selector.select_random_tasks(
            skills=getattr(args, 'skills', None),
            risks=getattr(args, 'risks', None),
            num_tasks=getattr(args, 'num_tasks', None),
            min_quality_score=getattr(args, 'min_quality_score', 2),
            seed=getattr(args, 'task_seed', None)
        )

        if not selected_tasks:
            print(f"❌ No tasks found matching criteria")
            sys.exit(1)

        # Override task_names with selected tasks
        args.task_names = selected_tasks
        print(f"\n🎯 Selected {len(selected_tasks)} tasks based on criteria:")
        for task in selected_tasks:
            print(f"   • {task}")
        print()

    # Handle optimization mode - test all hyperparameter combinations
    if hasattr(args, 'optimize') and args.optimize:
        from wisent.core.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig

        print(f"\n🔍 HYPERPARAMETER OPTIMIZATION MODE")
        print(f"   Model: {args.model}")
        print(f"   Task: {args.task_names}")

        # Load model first to detect layer count
        print(f"\n🤖 Loading model '{args.model}'...")
        model = WisentModel(args.model, device=args.device)
        num_layers = model.num_layers
        print(f"   ✓ Model loaded with {num_layers} layers")

        # Determine layer range
        if hasattr(args, 'optimize_layers') and args.optimize_layers == 'all':
            layer_range = list(range(num_layers))
        else:
            # Default: test ALL layers (0-indexed)
            layer_range = list(range(num_layers))

        # Create optimization config
        config = OptimizationConfig(
            layer_range=layer_range,
            aggregation_methods=['average', 'final', 'first', 'max'],
            prompt_construction_strategies=['multiple_choice', 'role_playing', 'direct_completion', 'instruction_following'],
            token_targeting_strategies=['choice_token', 'continuation_token', 'last_token', 'first_token', 'mean_pooling'],
            threshold_range=[0.3, 0.5, 0.7],
            classifier_types=[args.classifier_type],
            metric=getattr(args, 'optimize_metric', 'f1'),
            max_combinations=getattr(args, 'optimize_max_combinations', 1000)
        )

        total_combos = (len(layer_range) * len(config.aggregation_methods) *
                       len(config.prompt_construction_strategies) * len(config.token_targeting_strategies) *
                       len(config.threshold_range) * len(config.classifier_types))

        print(f"\n📊 Testing {total_combos} hyperparameter combinations:")
        print(f"   • Layers: {len(layer_range)} ({layer_range[0]}-{layer_range[-1]})")
        print(f"   • Aggregation methods: {len(config.aggregation_methods)}")
        print(f"   • Prompt strategies: {len(config.prompt_construction_strategies)}")
        print(f"   • Token strategies: {len(config.token_targeting_strategies)}")
        print(f"   • Thresholds: {len(config.threshold_range)}")
        print(f"   • Optimization metric: {config.metric}")

        # Load task data
        task_name = args.task_names[0] if isinstance(args.task_names, list) else args.task_names
        print(f"\n📊 Loading task '{task_name}'...")
        loader = LMEvalDataLoader()
        result = loader._load_one_task(
            task_name=task_name,
            split_ratio=args.split_ratio,
            seed=args.seed,
            limit=args.limit,
            training_limit=args.training_limit,
            testing_limit=args.testing_limit
        )
        train_pair_set = result['train_qa_pairs']
        test_pair_set = result['test_qa_pairs']
        print(f"   ✓ Loaded {len(train_pair_set.pairs)} training pairs, {len(test_pair_set.pairs)} test pairs")

        # Run optimization
        optimizer = HyperparameterOptimizer(config)
        opt_result = optimizer.optimize(
            model=model,
            train_pair_set=train_pair_set,
            test_pair_set=test_pair_set,
            device=args.device,
            verbose=True
        )

        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            results_path = os.path.join(args.output, 'optimization_results.json')
            optimizer.save_results(opt_result, results_path)
            print(f"\n💾 Optimization results saved to: {results_path}")

        print(f"\n✅ Optimization complete!")
        print(f"   Best layer: {opt_result.best_layer}")
        print(f"   Best aggregation: {opt_result.best_aggregation}")
        print(f"   Best prompt strategy: {opt_result.best_prompt_construction_strategy}")
        print(f"   Best token strategy: {opt_result.best_token_targeting_strategy}")
        print(f"   Best threshold: {opt_result.best_threshold}")
        print(f"   Best {config.metric}: {opt_result.best_score:.4f}")

        return {
            'best_hyperparameters': {
                'layer': opt_result.best_layer,
                'aggregation': opt_result.best_aggregation,
                'prompt_construction_strategy': opt_result.best_prompt_construction_strategy,
                'token_targeting_strategy': opt_result.best_token_targeting_strategy,
                'threshold': opt_result.best_threshold,
                'classifier_type': opt_result.best_classifier_type
            },
            'best_score': opt_result.best_score,
            'best_metrics': opt_result.best_metrics,
            'combinations_tested': len(opt_result.all_results)
        }

    # Handle cross-benchmark evaluation mode
    if hasattr(args, 'cross_benchmark') and args.cross_benchmark:
        if not (hasattr(args, 'train_task') and args.train_task and hasattr(args, 'eval_task') and args.eval_task):
            print(f"❌ Error: --cross-benchmark requires both --train-task and --eval-task")
            sys.exit(1)

        print(f"\n🎯 Cross-benchmark evaluation mode")
        print(f"   Training on: {args.train_task}")
        print(f"   Evaluating on: {args.eval_task}")
        print(f"   Model: {args.model}")
        print(f"   Layer: {args.layer}")
        print(f"   Classifier type: {args.classifier_type}")

        train_task_name = args.train_task
        eval_task_name = args.eval_task

        try:
            # Load training data directly (bypass split validation)
            print(f"\n📊 Loading training data from '{train_task_name}'...")
            from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
            from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

            loader = LMEvalDataLoader()
            train_task_obj = loader.load_lm_eval_task(train_task_name)

            # Handle both single task and group task (dict of tasks)
            if isinstance(train_task_obj, dict):
                # Group task - load all subtasks
                all_train_pairs = []
                training_limit_per_task = getattr(args, 'training_limit', None)
                if training_limit_per_task:
                    training_limit_per_task = training_limit_per_task // len(train_task_obj)

                for subname, task_obj in train_task_obj.items():
                    subtask_pairs = lm_build_contrastive_pairs(
                        task_name=subname,
                        lm_eval_task=task_obj,
                        limit=training_limit_per_task,
                    )
                    all_train_pairs.extend(subtask_pairs)
                train_pairs = all_train_pairs
            else:
                train_pairs = lm_build_contrastive_pairs(
                    task_name=train_task_name,
                    lm_eval_task=train_task_obj,
                    limit=getattr(args, 'training_limit', None),
                )

            train_pair_set = ContrastivePairSet("train", train_pairs, task_type=train_task_name)
            print(f"   ✓ Loaded {len(train_pair_set.pairs)} training pairs from {train_task_name}")

            # Load evaluation data directly (bypass split validation)
            print(f"\n📊 Loading evaluation data from '{eval_task_name}'...")
            eval_task_obj = loader.load_lm_eval_task(eval_task_name)

            # Handle both single task and group task (dict of tasks)
            if isinstance(eval_task_obj, dict):
                # Group task - load all subtasks
                all_eval_pairs = []
                testing_limit_per_task = getattr(args, 'testing_limit', None)
                if testing_limit_per_task:
                    testing_limit_per_task = testing_limit_per_task // len(eval_task_obj)

                for subname, task_obj in eval_task_obj.items():
                    subtask_pairs = lm_build_contrastive_pairs(
                        task_name=subname,
                        lm_eval_task=task_obj,
                        limit=testing_limit_per_task,
                    )
                    all_eval_pairs.extend(subtask_pairs)
                eval_pairs = all_eval_pairs
            else:
                eval_pairs = lm_build_contrastive_pairs(
                    task_name=eval_task_name,
                    lm_eval_task=eval_task_obj,
                    limit=getattr(args, 'testing_limit', None),
                )

            test_pair_set = ContrastivePairSet("test", eval_pairs, task_type=eval_task_name)
            print(f"   ✓ Loaded {len(test_pair_set.pairs)} test pairs from {eval_task_name}")

            # Override variables for rest of pipeline
            pair_set = train_pair_set
            task_name = train_task_name
            eval_task_name_for_later = eval_task_name

        except Exception as e:
            print(f"\n❌ Error loading cross-benchmark data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Standard single-task mode
        print(f"\n🎯 Starting classifier training on task: {args.task_names}")
        print(f"   Model: {args.model}")
        print(f"   Layer: {args.layer}")
        print(f"   Classifier type: {args.classifier_type}")

        try:
            # 1. Load task data using LMEvalDataLoader
            task_name = args.task_names[0] if isinstance(args.task_names, list) else args.task_names
            print(f"\n📊 Loading task '{task_name}'...")
            loader = LMEvalDataLoader()
            result = loader._load_one_task(
                task_name=task_name,
                split_ratio=args.split_ratio,
                seed=args.seed,
                limit=args.limit,
                training_limit=args.training_limit,
                testing_limit=args.testing_limit
            )

            # Use training pairs for classifier training
            pair_set = result['train_qa_pairs']
            test_pair_set = result['test_qa_pairs']
            eval_task_name_for_later = None
            print(f"   ✓ Loaded {len(pair_set.pairs)} training pairs")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 3. Load model
    print(f"\n🤖 Loading model '{args.model}'...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Model loaded with {model.num_layers} layers")

    # 3.5. Initialize detection handler if enabled
    detection_handler = None
    detection_stats = {
        'total_outputs': 0,
        'issues_detected': 0,
        'low_quality_outputs': 0,
        'handled_outputs': 0,
        'detection_types': {}
    }

    if hasattr(args, 'detection_action') and args.detection_action != 'pass_through':
        # Map string action to enum
        action_map = {
            'pass_through': DetectionAction.PASS_THROUGH,
            'replace_with_placeholder': DetectionAction.REPLACE_WITH_PLACEHOLDER,
            'regenerate_until_safe': DetectionAction.REGENERATE_UNTIL_SAFE
        }
        detection_action = action_map.get(args.detection_action, DetectionAction.REPLACE_WITH_PLACEHOLDER)

        detection_handler = DetectionHandler(
            action=detection_action,
            placeholder_message=getattr(args, 'placeholder_message', None),
            max_regeneration_attempts=getattr(args, 'max_regeneration_attempts', 3),
            log_detections=True
        )
        print(f"\n🛡️  Detection handling enabled")
        print(f"   Action: {args.detection_action}")
        print(f"   Threshold: {getattr(args, 'detection_threshold', 0.6)}")

    enable_quality_check = hasattr(args, 'enable_quality_check') and args.enable_quality_check
    if enable_quality_check:
        quality_threshold = getattr(args, 'quality_threshold', 50.0)
        print(f"\n🔍 Quality checking enabled")
        print(f"   Minimum quality score: {quality_threshold}/100")

    # 4. Parse layer specification
    layer = int(args.layer) if isinstance(args.layer, str) else args.layer
    print(f"\n🧠 Extracting activations from layer {layer}...")

    # 5. Collect activations for all pairs
    collector = ActivationCollector(model=model)

    # Get extraction strategy from args (already an ExtractionStrategy value string)
    extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', 'chat_last'))
    print(f"   Extraction strategy: {extraction_strategy.value}")

    positive_activations = []
    negative_activations = []

    # Convert layer int to string for activation collection
    layer_str = str(layer)

    for i, pair in enumerate(pair_set.pairs):
        if i % 10 == 0:
            print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...", end='\r')

        # Collect for positive (correct) response
        updated_pair = collector.collect(
            pair, strategy=extraction_strategy,
            layers=[layer_str],
        )

        # Extract activations from positive and negative responses
        if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
            act = updated_pair.positive_response.layers_activations[layer_str]
            if act is not None:
                # Convert to float32 before numpy (bfloat16 not supported by numpy)
                positive_activations.append(act.cpu().float().numpy())

        if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
            act = updated_pair.negative_response.layers_activations[layer_str]
            if act is not None:
                # Convert to float32 before numpy (bfloat16 not supported by numpy)
                negative_activations.append(act.cpu().float().numpy())

    print(f"\n   ✓ Collected {len(positive_activations)} positive and {len(negative_activations)} negative activations")

    # Check if steering vector mode is requested
    if args.save_steering_vector and args.train_only:
        import torch
        from wisent.core.steering_methods.methods.caa import CAAMethod

        print(f"\n🎯 Training steering vector using {args.steering_method} method...")

        # Convert activations to tensors
        pos_tensors = [torch.from_numpy(act).float() for act in positive_activations]
        neg_tensors = [torch.from_numpy(act).float() for act in negative_activations]

        # Create steering method
        steering_method = CAAMethod(normalize=True)

        # Train steering vector
        steering_vector = steering_method.train_for_layer(pos_tensors, neg_tensors)

        # Save steering vector
        print(f"\n💾 Saving steering vector to '{args.save_steering_vector}'...")
        os.makedirs(os.path.dirname(args.save_steering_vector) or '.', exist_ok=True)
        torch.save({
            'steering_vector': steering_vector,
            'layer_index': layer,
            'method': args.steering_method,
            'model': args.model,
            'task': args.task_names,
            # Legacy keys for backward compatibility
            'vector': steering_vector,
            'layer': layer,
        }, args.save_steering_vector)
        print(f"   ✓ Steering vector saved to: {args.save_steering_vector}")

        # Save output artifacts if requested
        if args.output:
            print(f"\n📁 Saving artifacts to '{args.output}'...")
            os.makedirs(args.output, exist_ok=True)
            report_path = os.path.join(args.output, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump({
                    'method': args.steering_method,
                    'layer': layer,
                    'num_positive': len(positive_activations),
                    'num_negative': len(negative_activations),
                    'vector_shape': list(steering_vector.shape)
                }, f, indent=2)
            print(f"   ✓ Training report saved to: {report_path}")

        print(f"\n✅ Steering vector training completed successfully!\n")

        # Return results for programmatic access
        return {
            "steering_vector_saved": True,
            "vector_path": args.save_steering_vector,
            "layer": layer,
            "method": args.steering_method,
            "num_positive": len(positive_activations),
            "num_negative": len(negative_activations),
            "vector_shape": list(steering_vector.shape),
            "test_accuracy": None,
            "test_f1_score": None,
            "training_time": 0.0,
            "evaluation_results": {}
        }

    # 6. Prepare training data
    print(f"\n🎯 Preparing training data...")
    X_positive = np.array(positive_activations)
    X_negative = np.array(negative_activations)
    X = np.vstack([X_positive, X_negative])
    y = np.array([1] * len(positive_activations) + [0] * len(negative_activations))

    print(f"   Training set: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Positive samples: {sum(y == 1)}, Negative samples: {sum(y == 0)}")

    # 7. Create and train classifier
    print(f"\n🏋️  Training {args.classifier_type} classifier...")
    if args.classifier_type == 'logistic':
        classifier = LogisticClassifier(threshold=args.detection_threshold, device=args.device)
    elif args.classifier_type == 'mlp':
        classifier = MLPClassifier(threshold=args.detection_threshold, device=args.device)
    else:
        raise UnknownTypeError(entity_type="classifier_type", value=args.classifier_type, valid_values=["logistic", "mlp"])

    # Training configuration
    train_config = ClassifierTrainConfig(
        test_size=1.0 - args.split_ratio,
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        monitor='f1',
        random_state=args.seed
    )

    # Train the classifier
    report = classifier.fit(X, y, config=train_config)

    # 8. Print training completion
    print(f"\n📈 Training completed!")
    print(f"   Best epoch: {report.best_epoch}/{report.epochs_ran}")

    # 8.5. PROPER EVALUATION: Test classifier on real model generations
    print(f"\n🎯 Evaluating classifier on real model generations...")

    # Get test pairs (already set from cross-benchmark or single-task mode)
    test_pairs = test_pair_set
    eval_task_for_evaluator = eval_task_name_for_later if eval_task_name_for_later else task_name

    if eval_task_name_for_later:
        print(f"   Cross-benchmark: Testing on {eval_task_name_for_later} (trained on {task_name})")

    print(f"   Generating responses for {len(test_pairs.pairs)} test questions...")

    # Initialize evaluator for this task
    from wisent.core.evaluators.rotator import EvaluatorRotator
    # Discover both oracles and benchmark_specific evaluators
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=eval_task_for_evaluator, autoload=False)
    print(f"   Using evaluator: coding")

    # Generate responses and collect activations
    generation_results = []
    for i, pair in enumerate(test_pairs.pairs):
        if i % 10 == 0:
            print(f"      Processing {i+1}/{len(test_pairs.pairs)}...", end='\r')

        question = pair.prompt
        expected = pair.positive_response.model_response
        choices = [pair.negative_response.model_response, pair.positive_response.model_response]

        # Generate response from unsteered model
        messages = [{"role": "user", "content": question}]

        response = model.generate(
            [messages],
            **get_generate_kwargs(),
        )[0]

        # Evaluate the response using Wisent evaluator
        # Pass all pair metadata to evaluator - each evaluator uses what it needs
        eval_kwargs = {
            'response': response,
            'expected': expected,
            'model': model,
            'question': question,
            'choices': choices,
            'task_name': task_name,
        }
        # Add all pair metadata to eval_kwargs (test_code, correct_answers, etc.)
        if hasattr(pair, 'metadata') and pair.metadata:
            for key, value in pair.metadata.items():
                if value is not None and key not in eval_kwargs:
                    eval_kwargs[key] = value
        eval_result = evaluator.evaluate(**eval_kwargs)

        # Get activation for this generation
        # Use ActivationCollector to collect activations from the generated text
        gen_collector = ActivationCollector(model=model)
        # Create a pair with the generated response
        from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
        from wisent.core.contrastive_pairs.core.pair import ContrastivePair
        temp_pos_response = PositiveResponse(model_response=response, layers_activations={})
        temp_neg_response = NegativeResponse(model_response="placeholder", layers_activations={})  # Not used
        temp_pair = ContrastivePair(
            prompt=question,
            positive_response=temp_pos_response,
            negative_response=temp_neg_response,
            label=None,
            trait_description=None
        )

        # Collect activation - ActivationCollector will re-run the model with prompt+response
        # First, collect with full sequence to get token-by-token activations
        collected_full = gen_collector.collect(
            temp_pair, strategy=extraction_strategy,
        )

        # Access the collected activations
        import torch
        if collected_full.positive_response.layers_activations:
            layer_activations = collected_full.positive_response.layers_activations
            if layer_str in layer_activations:
                activation = layer_activations[layer_str]
                if activation is not None and isinstance(activation, torch.Tensor):
                    # activation shape: (hidden_dim,) - already aggregated by extraction strategy
                    act_tensor = activation.unsqueeze(0).float()
                    pred_proba_result = classifier.predict_proba(act_tensor)
                    pred_proba = pred_proba_result if isinstance(pred_proba_result, float) else pred_proba_result[0]
                    pred_label = int(pred_proba > args.detection_threshold)

                    # Update detection stats
                    detection_stats['total_outputs'] += 1
                    original_response = response
                    quality_score = None
                    issue_detected = False
                    detection_type = None

                    # Quality check: detect gibberish/low-quality outputs
                    if enable_quality_check:
                        import torch as quality_torch
                        # Get device from model
                        model_device = quality_torch.device(model.device)
                        quality_score = evaluate_quality(
                            response=response,
                            model=model.hf_model,  # WisentModel wraps the HF model
                            tokenizer=model.tokenizer,
                            device=model_device
                        )

                        if quality_score < quality_threshold:
                            detection_stats['low_quality_outputs'] += 1
                            issue_detected = True
                            detection_type = 'low_quality'

                            if detection_handler:
                                # Handle low quality output
                                response = detection_handler.handle_detection(
                                    original_response=response,
                                    detection_type='gibberish',
                                    confidence_score=(quality_threshold - quality_score) / quality_threshold,
                                    original_prompt=question,
                                    regenerate_function=None  # Not implemented for now
                                )
                                detection_stats['handled_outputs'] += 1

                    # Classifier detection: handle when classifier detects issue
                    if detection_handler and pred_label == 0 and pred_proba < args.detection_threshold:
                        # Classifier detected negative/problematic content
                        detection_stats['issues_detected'] += 1
                        issue_detected = True
                        detection_type = 'classifier_flagged'

                        # Determine detection type based on task
                        if 'halluc' in task_name.lower():
                            detection_type = 'hallucination'
                        elif 'bias' in task_name.lower():
                            detection_type = 'bias'
                        elif 'harm' in task_name.lower():
                            detection_type = 'harmful_content'
                        else:
                            detection_type = 'unknown_issue'

                        # Track detection types
                        if detection_type not in detection_stats['detection_types']:
                            detection_stats['detection_types'][detection_type] = 0
                        detection_stats['detection_types'][detection_type] += 1

                        # Handle the detection
                        response = detection_handler.handle_detection(
                            original_response=response,
                            detection_type=detection_type,
                            confidence_score=float(1.0 - pred_proba),
                            original_prompt=question,
                            regenerate_function=None  # Not implemented for now
                        )
                        detection_stats['handled_outputs'] += 1

                    # Ground truth from evaluator
                    ground_truth = 1 if eval_result.ground_truth == "TRUTHFUL" else 0

                    generation_results.append({
                        'question': question,
                        'response': response,
                        'original_response': original_response,
                        'expected': expected,
                        'eval_result': eval_result.ground_truth,
                        'classifier_pred': pred_label,
                        'classifier_proba': float(pred_proba),
                        'correct': pred_label == ground_truth,
                        'quality_score': quality_score,
                        'issue_detected': issue_detected,
                        'detection_type': detection_type,
                        'response_modified': response != original_response
                    })

    print(f"\n   ✓ Evaluated {len(generation_results)} generations")

    # Calculate real-world metrics
    if generation_results:
        correct_predictions = sum(1 for r in generation_results if r['correct'])
        real_accuracy = correct_predictions / len(generation_results)

        # Calculate precision, recall, F1 on real generations
        true_positives = sum(1 for r in generation_results if r['classifier_pred'] == 1 and r['eval_result'] == 'TRUTHFUL')
        false_positives = sum(1 for r in generation_results if r['classifier_pred'] == 1 and r['eval_result'] == 'UNTRUTHFUL')
        false_negatives = sum(1 for r in generation_results if r['classifier_pred'] == 0 and r['eval_result'] == 'TRUTHFUL')

        real_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        real_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0

        print(f"\n   📊 Real-world performance (on actual generations):")
        print(f"     • Accuracy:  {real_accuracy:.4f}")
        print(f"     • Precision: {real_precision:.4f}")
        print(f"     • Recall:    {real_recall:.4f}")
        print(f"     • F1 Score:  {real_f1:.4f}")

        # Print quality statistics if quality checking was enabled
        if enable_quality_check:
            quality_scores = [r['quality_score'] for r in generation_results if r['quality_score'] is not None]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)
                print(f"\n   📊 Quality Statistics:")
                print(f"     • Average quality: {avg_quality:.2f}/100")
                print(f"     • Min quality:     {min_quality:.2f}/100")
                print(f"     • Max quality:     {max_quality:.2f}/100")
                print(f"     • Threshold:       {quality_threshold:.2f}/100")

        # Print detection statistics
        if detection_handler or enable_quality_check:
            print(f"\n   🛡️  Detection Statistics:")
            print(f"     • Total outputs:        {detection_stats['total_outputs']}")
            print(f"     • Issues detected:      {detection_stats['issues_detected']}")
            print(f"     • Low quality outputs:  {detection_stats['low_quality_outputs']}")
            print(f"     • Handled outputs:      {detection_stats['handled_outputs']}")

            if detection_stats['detection_types']:
                print(f"     • Detection breakdown:")
                for det_type, count in detection_stats['detection_types'].items():
                    print(f"       - {det_type}: {count}")

            # Calculate percentage of outputs that had issues
            if detection_stats['total_outputs'] > 0:
                issue_rate = (detection_stats['issues_detected'] + detection_stats['low_quality_outputs']) / detection_stats['total_outputs']
                print(f"     • Issue rate:           {issue_rate:.1%}")

    else:
        real_accuracy = real_f1 = real_precision = real_recall = 0.0
        generation_results = []

    # 9. Save classifier if requested
    if args.save_classifier:
        print(f"\n💾 Saving classifier to '{args.save_classifier}'...")

        # Create metadata
        metadata = create_classifier_metadata(
            model_name=args.model,
            task_name=args.task_names,
            layer=layer,
            classifier_type=args.classifier_type,
            training_accuracy=report.final.accuracy,
            training_samples=len(X),
            token_aggregation=extraction_strategy.value,
            detection_threshold=args.detection_threshold
        )

        # Save using model persistence
        save_path = ModelPersistence.save_classifier(
            classifier=classifier,
            layer=layer,
            save_path=args.save_classifier,
            metadata=metadata
        )
        print(f"   ✓ Classifier saved to: {save_path}")

    # 10. Save output artifacts if requested
    if args.output:
        print(f"\n📁 Saving artifacts to '{args.output}'...")
        os.makedirs(args.output, exist_ok=True)

        # Save training report
        report_path = os.path.join(args.output, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report.asdict(), f, indent=2)
        print(f"   ✓ Training report saved to: {report_path}")

        # Save generation details with token scores
        if generation_results:
            generation_path = os.path.join(args.output, 'generation_details.json')
            with open(generation_path, 'w') as f:
                json.dump({
                    'task': args.task_names,
                    'model': args.model,
                    'layer': layer,
                    'aggregation': extraction_strategy.value,
                    'threshold': args.detection_threshold,
                    'num_generations': len(generation_results),
                    'detection_stats': detection_stats,
                    'quality_check_enabled': enable_quality_check,
                    'quality_threshold': quality_threshold if enable_quality_check else None,
                    'generations': generation_results
                }, f, indent=2)
            print(f"   ✓ Generation details (with token scores) saved to: {generation_path}")

    print(f"\n✅ Task completed successfully!\n")

    # Return results for programmatic access
    return {
        # Real-world metrics (on actual generations) - THE ONLY METRICS THAT MATTER
        "accuracy": float(real_accuracy),
        "f1_score": float(real_f1),
        "precision": float(real_precision),
        "recall": float(real_recall),
        "generation_count": len(generation_results),
        # Metadata
        "best_epoch": report.best_epoch,
        "epochs_ran": report.epochs_ran,
        "generation_details": generation_results
    }
