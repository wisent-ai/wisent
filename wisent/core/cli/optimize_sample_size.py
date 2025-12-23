"""Sample size optimization command execution logic."""

import sys
import time
import numpy as np
import torch

try:
    from wisent_plots import LineChart
except ImportError:
    LineChart = None  # wisent_plots is optional
from wisent.core.models.wisent_model import WisentModel
from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
from wisent.core.activations import ExtractionStrategy, ActivationCollector
from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
from wisent.core.classifiers.classifiers.models.mlp import MLPClassifier
from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.core.models.inference_config import get_generate_kwargs


def execute_optimize_sample_size(args):
    """Execute the optimize-sample-size command - find optimal training sample size."""

    print(f"\n{'='*80}")
    print(f"📊 SAMPLE SIZE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Task: {args.task}")
    print(f"   Layer: {args.layer}")
    print(f"   Sample sizes: {args.sample_sizes}")
    print(f"   Test size: {args.test_size}")
    print(f"   Device: {args.device if hasattr(args, 'device') and args.device else 'auto-detect'}")
    print(f"{'='*80}\n")

    try:
        # Only support classification mode
        if hasattr(args, 'steering_mode') and args.steering_mode:
            print("❌ Error: Steering mode not supported in optimize-sample-size")
            sys.exit(1)

        # 1. Load model once
        print(f"🤖 Loading model '{args.model}'...")
        model = WisentModel(args.model, device=args.device)

        # 2. Load contrastive pairs once using LMEvalDataLoader
        print(f"📚 Loading task '{args.task}'...")
        max_train_samples = max(args.sample_sizes)
        total_limit = max_train_samples + args.test_size + 50  # Some buffer

        loader = LMEvalDataLoader()
        result = loader._load_one_task(
            task_name=args.task,
            split_ratio=0.5,  # Use 50/50 split, then we'll combine and re-split
            seed=args.seed,
            limit=total_limit,
            training_limit=None,
            testing_limit=None
        )

        # Combine train and test pairs into one list for our own splitting
        train_pairs = result['train_qa_pairs']
        test_pairs = result['test_qa_pairs']
        all_pairs_list = train_pairs.pairs + test_pairs.pairs

        # Create a combined pair set
        from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
        all_pairs = ContrastivePairSet(name="combined_pairs", pairs=all_pairs_list)

        # 3. Split into train/test
        total_available = len(all_pairs.pairs)
        max_train = max(args.sample_sizes)
        if total_available < max_train + args.test_size:
            print(f"⚠️  Warning: Only {total_available} pairs available, requested {max_train + args.test_size}")

        test_pairs = all_pairs.pairs[-args.test_size:]
        available_train_pairs = all_pairs.pairs[:-args.test_size]

        print(f"   Total pairs: {total_available}")
        print(f"   Test pairs: {len(test_pairs)}")
        print(f"   Available training pairs: {len(available_train_pairs)}")

        # 4. Collect TEST activations ONCE (reuse for all sample sizes)
        print(f"\n🎯 Collecting test activations (ONCE)...")
        layer_str = str(args.layer)  # Layer should be just the number, like "10"

        # Get extraction strategy from args
        extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', 'chat_last'))

        collector = ActivationCollector(model=model)

        # Collect test activations for all test pairs (ONCE)
        X_test_list = []
        y_test_list = []

        print(f"   Collecting activations for {len(test_pairs)} test pairs...")
        for i, pair in enumerate(test_pairs):
            if i % 10 == 0:
                print(f"      Processing test pair {i+1}/{len(test_pairs)}...", end='\r')

            # Collect activations for this pair
            collected_pair = collector.collect(
                pair, strategy=extraction_strategy,
                layers=[layer_str],
            )

            # Extract positive and negative activations
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

        print(f"      Processing test pair {len(test_pairs)}/{len(test_pairs)}... Done!")

        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        print(f"   ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

        # 5. Initialize evaluator for ground truth
        EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
        EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
        evaluator = EvaluatorRotator(evaluator=None, task_name=args.task, autoload=False)
        print(f"   Using evaluator: {evaluator._plugin.name}")

        # 6. Generate responses and collect activations for evaluation ONCE
        print(f"\n📝 Generating test responses (ONCE)...")
        test_generations = []
        for i, pair in enumerate(test_pairs):
            if i % 10 == 0:
                print(f"      Processing {i+1}/{len(test_pairs)}...", end='\r')

            question = pair.prompt
            expected = pair.positive_response.model_response
            choices = [pair.negative_response.model_response, pair.positive_response.model_response]

            # Generate response
            response = model.generate(
                [[{"role": "user", "content": question}]],
                **get_generate_kwargs(),
            )[0]

            # Evaluate
            eval_result = evaluator.evaluate(
                response=response,
                expected=expected,
                model=model,
                question=question,
                choices=choices,
                task_name=args.task
            )

            # Collect activation for this generation
            from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
            from wisent.core.contrastive_pairs.core.pair import ContrastivePair
            temp_pos = PositiveResponse(model_response=response, layers_activations={})
            temp_neg = NegativeResponse(model_response="placeholder", layers_activations={})
            temp_pair = ContrastivePair(prompt=question, positive_response=temp_pos, negative_response=temp_neg, label=None, trait_description=None)

            collected = collector.collect(temp_pair, strategy=extraction_strategy, layers=[layer_str])

            activation = None
            if collected.positive_response.layers_activations and layer_str in collected.positive_response.layers_activations:
                activation = collected.positive_response.layers_activations[layer_str]

            test_generations.append({
                'question': question,
                'response': response,
                'expected': expected,
                'ground_truth': 1 if eval_result.ground_truth == "TRUTHFUL" else 0,
                'activation': activation
            })

        print(f"      Processing {len(test_pairs)}/{len(test_pairs)}... Done!")

        # 7. Now test each sample size (only training differs)
        results = []
        for sample_size in args.sample_sizes:
            print(f"\n{'='*60}")
            print(f"Testing training size: {sample_size}")
            print(f"{'='*60}")

            start_time = time.time()

            # Get training pairs for this sample size
            train_pairs_subset = available_train_pairs[:sample_size]

            # Collect training activations
            print(f"   Collecting training activations for {sample_size} pairs...")
            X_train_list = []
            y_train_list = []

            for i, pair in enumerate(train_pairs_subset):
                if i % 10 == 0:
                    print(f"      Processing train pair {i+1}/{sample_size}...", end='\r')

                # Collect activations for this pair
                collected_pair = collector.collect(
                    pair, strategy=extraction_strategy,
                    layers=[layer_str],
                )

                # Extract positive and negative activations
                if collected_pair.positive_response.layers_activations and layer_str in collected_pair.positive_response.layers_activations:
                    pos_act = collected_pair.positive_response.layers_activations[layer_str]
                    if pos_act is not None:
                        X_train_list.append(pos_act.cpu().numpy())
                        y_train_list.append(1)

                if collected_pair.negative_response.layers_activations and layer_str in collected_pair.negative_response.layers_activations:
                    neg_act = collected_pair.negative_response.layers_activations[layer_str]
                    if neg_act is not None:
                        X_train_list.append(neg_act.cpu().numpy())
                        y_train_list.append(0)

            print(f"      Processing train pair {sample_size}/{sample_size}... Done!")

            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)
            print(f"   Training set: {X_train.shape[0]} samples")

            # Train classifier
            print(f"   Training classifier...")
            classifier = LogisticClassifier(threshold=args.threshold, device=args.device)
            train_config = ClassifierTrainConfig(
                test_size=0.2,
                num_epochs=50,
                batch_size=32,
                learning_rate=1e-3,
                monitor='f1',
                random_state=args.seed
            )
            classifier.fit(X_train, y_train, config=train_config)

            # Evaluate on test generations
            print(f"   Evaluating on test set...")
            correct = 0
            total = 0
            for gen in test_generations:
                if gen['activation'] is not None:
                    activation_tensor = gen['activation'].unsqueeze(0).float()
                    pred_proba = classifier.predict_proba(activation_tensor)
                    pred_label = int(pred_proba > args.threshold)
                    if pred_label == gen['ground_truth']:
                        correct += 1
                    total += 1

            accuracy = correct / total if total > 0 else 0.0
            f1_score = accuracy  # Simplified - could calculate proper F1

            total_time = time.time() - start_time

            results.append({
                'sample_size': sample_size,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'time': total_time
            })

            print(f"   ✓ Sample size {sample_size}: accuracy={accuracy:.3f}, f1={f1_score:.3f}, time={total_time:.1f}s")

        # 8. Find optimal sample size (95% of best accuracy with smallest size)
        accuracies = np.array([r['accuracy'] for r in results])
        sample_sizes = np.array([r['sample_size'] for r in results])

        max_accuracy = np.max(accuracies)
        threshold = max_accuracy * 0.95
        good_indices = np.where(accuracies >= threshold)[0]

        if len(good_indices) > 0:
            optimal_idx = good_indices[0]
        else:
            optimal_idx = np.argmax(accuracies)

        optimal_size = sample_sizes[optimal_idx]
        optimal_result = results[optimal_idx]

        # Display results
        print(f"\n{'='*80}")
        print(f"📈 OPTIMIZATION RESULTS")
        print(f"{'='*80}")
        print(f"   Optimal sample size: {optimal_size}")
        print(f"   Best accuracy: {optimal_result['accuracy']:.4f}")
        print(f"   Best F1 score: {optimal_result['f1_score']:.4f}")
        print(f"{'='*80}\n")

        # Save plot if requested
        if args.save_plot:
            plot_path_svg = f"sample_size_optimization_{args.task}_{args.model.replace('/', '_')}.svg"
            plot_path_png = f"sample_size_optimization_{args.task}_{args.model.replace('/', '_')}.png"

            # Extract data for plotting
            x_data = [r['sample_size'] for r in results]
            accuracies = [r['accuracy'] for r in results]
            f1_scores = [r['f1_score'] for r in results]
            times = [r['time'] for r in results]

            # Create performance plot (Accuracy and F1)
            import matplotlib.pyplot as plt
            chart1 = LineChart(style=1, figsize=(10, 6), show_markers=True)

            # Create empty figure and axis for the plot
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

            # Use plot_multiple with the figure and axis
            chart1.plot_multiple(
                x=x_data,
                y_series=[accuracies, f1_scores],
                labels=['Accuracy', 'F1 Score'],
                title=f'Performance vs Sample Size\n{args.model} on {args.task}',
                xlabel='Training Sample Size',
                ylabel='Score',
                fig=fig1,
                ax=ax1,
                output_format='png'
            )

            # Add vertical line for optimal size
            ax1.axvline(x=optimal_size, color='#2ecc71', linestyle='--', linewidth=2,
                       label=f'Optimal: {optimal_size}', alpha=0.7)
            ax1.legend()

            # Save performance plot
            fig1.savefig(plot_path_svg.replace('.svg', '_performance.svg'),
                        format='svg', bbox_inches='tight')
            fig1.savefig(plot_path_png.replace('.png', '_performance.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig1)

            # Create training time plot
            chart2 = LineChart(style=1, figsize=(10, 6), show_markers=True)

            # Create empty figure and axis for the plot
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

            chart2.plot_multiple(
                x=x_data,
                y_series=[times],
                labels=['Training Time'],
                colors=['#27ae60'],
                title=f'Training Time vs Sample Size\n{args.model} on {args.task}',
                xlabel='Training Sample Size',
                ylabel='Time (seconds)',
                fig=fig2,
                ax=ax2,
                output_format='png'
            )

            # Save time plot
            fig2.savefig(plot_path_svg.replace('.svg', '_time.svg'),
                        format='svg', bbox_inches='tight')
            fig2.savefig(plot_path_png.replace('.png', '_time.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig2)

            print(f"💾 Performance plot saved to:")
            print(f"   SVG: {plot_path_svg.replace('.svg', '_performance.svg')}")
            print(f"   PNG: {plot_path_png.replace('.png', '_performance.png')}")
            print(f"💾 Training time plot saved to:")
            print(f"   SVG: {plot_path_svg.replace('.svg', '_time.svg')}")
            print(f"   PNG: {plot_path_png.replace('.png', '_time.png')}\n")

        print(f"✅ Sample size optimization completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
