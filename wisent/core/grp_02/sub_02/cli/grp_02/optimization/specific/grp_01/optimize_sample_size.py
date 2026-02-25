"""Sample size optimization command execution logic."""

import sys
import time
import numpy as np
import torch

from wisent.core.models.wisent_model import WisentModel
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.activations import ExtractionStrategy, ActivationCollector
from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.core.models import get_generate_kwargs
from wisent.core.constants import DEFAULT_CLASSIFIER_LR, CLASSIFIER_TEST_SIZE, CLASSIFIER_NUM_EPOCHS, CLASSIFIER_BATCH_SIZE, OPTIMIZE_ACCURACY_THRESHOLD_MULT, AUTOTUNE_VAL_SPLIT, SAMPLE_LOADING_BUFFER, PROGRESS_LOG_INTERVAL_10, SEPARATOR_WIDTH_STANDARD


def execute_optimize_sample_size(args):
    """Execute the optimize-sample-size command - find optimal training sample size."""

    print(f"\n{'='*80}")
    print(f"SAMPLE SIZE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Task: {args.task}")
    print(f"   Layer: {args.layer}")
    print(f"   Sample sizes: {args.sample_sizes}")
    print(f"   Test size: {args.test_size}")
    print(f"   Device: {args.device if hasattr(args, 'device') and args.device else 'auto-detect'}")
    print(f"{'='*80}\n")

    try:
        if hasattr(args, 'steering_mode') and args.steering_mode:
            print("Error: Steering mode not supported in optimize-sample-size")
            sys.exit(1)

        print(f"Loading model '{args.model}'...")
        model = WisentModel(args.model, device=args.device)

        print(f"Loading task '{args.task}'...")
        max_train_samples = max(args.sample_sizes)
        total_limit = max_train_samples + args.test_size + SAMPLE_LOADING_BUFFER

        loader = LMEvalDataLoader()
        result = loader._load_one_task(
            task_name=args.task, split_ratio=AUTOTUNE_VAL_SPLIT, seed=args.seed,
            limit=total_limit, training_limit=None, testing_limit=None
        )

        train_pairs = result['train_qa_pairs']
        test_pairs = result['test_qa_pairs']
        all_pairs_list = train_pairs.pairs + test_pairs.pairs

        from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
        all_pairs = ContrastivePairSet(name="combined_pairs", pairs=all_pairs_list)

        total_available = len(all_pairs.pairs)
        max_train = max(args.sample_sizes)
        if total_available < max_train + args.test_size:
            print(f"Warning: Only {total_available} pairs available, requested {max_train + args.test_size}")

        test_pairs = all_pairs.pairs[-args.test_size:]
        available_train_pairs = all_pairs.pairs[:-args.test_size]

        print(f"   Total pairs: {total_available}")
        print(f"   Test pairs: {len(test_pairs)}")
        print(f"   Available training pairs: {len(available_train_pairs)}")

        print(f"\nCollecting test activations (ONCE)...")
        layer_str = str(args.layer)
        extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', 'chat_last'))
        collector = ActivationCollector(model=model)

        X_test_list, y_test_list = [], []
        print(f"   Collecting activations for {len(test_pairs)} test pairs...")
        for i, pair in enumerate(test_pairs):
            if i % PROGRESS_LOG_INTERVAL_10 == 0:
                print(f"      Processing test pair {i+1}/{len(test_pairs)}...", end='\r')
            collected_pair = collector.collect(pair, strategy=extraction_strategy, layers=[layer_str])
            if collected_pair.positive_response.layers_activations and layer_str in collected_pair.positive_response.layers_activations:
                pos_act = collected_pair.positive_response.layers_activations[layer_str]
                if pos_act is not None:
                    X_test_list.append(pos_act.cpu().numpy())
                    y_test_list.append(1)
            if collected_pair.negative_response.layers_activations and layer_str in collected_pair.negative_response.layers_activations:
                neg_act = collected_pair.negative_response.layers_activations[layer_str]
                if neg_act is not None:
                    X_test_list.append(neg_act.cpu().numpy())
                    y_test_list.append(0)

        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        print(f"\n   Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

        EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
        EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
        evaluator = EvaluatorRotator(evaluator=None, task_name=args.task, autoload=False)
        print(f"   Using evaluator: {evaluator._plugin.name}")

        print(f"\nGenerating test responses (ONCE)...")
        test_generations = []
        for i, pair in enumerate(test_pairs):
            if i % PROGRESS_LOG_INTERVAL_10 == 0:
                print(f"      Processing {i+1}/{len(test_pairs)}...", end='\r')
            question = pair.prompt
            expected = pair.positive_response.model_response
            choices = [pair.negative_response.model_response, pair.positive_response.model_response]
            response = model.generate([[{"role": "user", "content": question}]], **get_generate_kwargs())[0]
            eval_result = evaluator.evaluate(response=response, expected=expected, model=model,
                                             question=question, choices=choices, task_name=args.task)
            from wisent.core.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
            from wisent.core.contrastive_pairs.core.pair import ContrastivePair
            temp_pos = PositiveResponse(model_response=response, layers_activations={})
            temp_neg = NegativeResponse(model_response="placeholder", layers_activations={})
            temp_pair = ContrastivePair(prompt=question, positive_response=temp_pos, negative_response=temp_neg, label=None, trait_description=None)
            collected = collector.collect(temp_pair, strategy=extraction_strategy, layers=[layer_str])
            activation = None
            if collected.positive_response.layers_activations and layer_str in collected.positive_response.layers_activations:
                activation = collected.positive_response.layers_activations[layer_str]
            test_generations.append({
                'question': question, 'response': response, 'expected': expected,
                'ground_truth': 1 if eval_result.ground_truth == "TRUTHFUL" else 0, 'activation': activation
            })

        # Test each sample size
        results = []
        for sample_size in args.sample_sizes:
            print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}\nTesting training size: {sample_size}\n{'='*SEPARATOR_WIDTH_STANDARD}")
            start_time = time.time()
            train_pairs_subset = available_train_pairs[:sample_size]
            X_train_list, y_train_list = [], []
            for i, pair in enumerate(train_pairs_subset):
                if i % PROGRESS_LOG_INTERVAL_10 == 0:
                    print(f"      Processing train pair {i+1}/{sample_size}...", end='\r')
                collected_pair = collector.collect(pair, strategy=extraction_strategy, layers=[layer_str])
                if collected_pair.positive_response.layers_activations and layer_str in collected_pair.positive_response.layers_activations:
                    pos_act = collected_pair.positive_response.layers_activations[layer_str]
                    if pos_act is not None:
                        X_train_list.append(pos_act.cpu().numpy()); y_train_list.append(1)
                if collected_pair.negative_response.layers_activations and layer_str in collected_pair.negative_response.layers_activations:
                    neg_act = collected_pair.negative_response.layers_activations[layer_str]
                    if neg_act is not None:
                        X_train_list.append(neg_act.cpu().numpy()); y_train_list.append(0)

            X_train = np.array(X_train_list); y_train = np.array(y_train_list)
            classifier = LogisticClassifier(threshold=args.threshold, device=args.device)
            train_config = ClassifierTrainConfig(test_size=CLASSIFIER_TEST_SIZE, num_epochs=CLASSIFIER_NUM_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE, learning_rate=DEFAULT_CLASSIFIER_LR, monitor='f1', random_state=args.seed)
            classifier.fit(X_train, y_train, config=train_config)

            correct, total = 0, 0
            for gen in test_generations:
                if gen['activation'] is not None:
                    pred_proba = classifier.predict_proba(gen['activation'].unsqueeze(0).float())
                    if int(pred_proba > args.threshold) == gen['ground_truth']:
                        correct += 1
                    total += 1
            accuracy = correct / total if total > 0 else 0.0
            total_time = time.time() - start_time
            results.append({'sample_size': sample_size, 'accuracy': accuracy, 'f1_score': accuracy, 'time': total_time})
            print(f"\n   Sample size {sample_size}: accuracy={accuracy:.3f}, time={total_time:.1f}s")

        # Find optimal sample size
        accuracies = np.array([r['accuracy'] for r in results])
        sample_sizes = np.array([r['sample_size'] for r in results])
        max_accuracy = np.max(accuracies)
        threshold = max_accuracy * OPTIMIZE_ACCURACY_THRESHOLD_MULT
        good_indices = np.where(accuracies >= threshold)[0]
        optimal_idx = good_indices[0] if len(good_indices) > 0 else np.argmax(accuracies)
        optimal_size = sample_sizes[optimal_idx]
        optimal_result = results[optimal_idx]

        print(f"\n{'='*80}\nOPTIMIZATION RESULTS\n{'='*80}")
        print(f"   Optimal sample size: {optimal_size}")
        print(f"   Best accuracy: {optimal_result['accuracy']:.4f}")
        print(f"{'='*80}\n")

        if args.save_plot:
            from wisent.core.cli.optimization.specific._helpers.optimize_sample_size_helpers import save_optimization_plots
            save_optimization_plots(args, results, optimal_size)

        print(f"Sample size optimization completed successfully!\n")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
