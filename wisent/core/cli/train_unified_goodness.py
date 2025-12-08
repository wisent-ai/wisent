"""Train unified goodness vector from pooled multi-benchmark data."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


def load_all_benchmarks():
    """
    Load ALL benchmarks from the parameter files.
    
    This uses the same benchmark list as test_all_benchmarks.py:
    - all_lm_eval_task_families.json (165 task families)
    - not_lm_eval_tasks.json (170 additional tasks)
    - minus broken_in_lm_eval.json (8 broken)
    
    Total: 327 usable benchmarks
    """
    # Find the parameters directory
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    
    # Load lm-eval task families
    lm_eval_tasks_path = params_dir / "all_lm_eval_task_families.json"
    with open(lm_eval_tasks_path, 'r') as f:
        lm_eval_tasks = json.load(f)
    
    # Load non lm-eval tasks
    not_lm_eval_tasks_path = params_dir / "not_lm_eval_tasks.json"
    with open(not_lm_eval_tasks_path, 'r') as f:
        not_lm_eval_tasks = json.load(f)
    
    # Load broken benchmarks to skip
    broken_tasks_path = params_dir / "broken_in_lm_eval.json"
    broken_tasks = []
    if broken_tasks_path.exists():
        with open(broken_tasks_path, 'r') as f:
            broken_tasks = json.load(f)
    
    # Combine all tasks and filter out broken ones
    all_tasks = lm_eval_tasks + not_lm_eval_tasks
    filtered_tasks = [task for task in all_tasks if task not in broken_tasks]
    
    return filtered_tasks, broken_tasks


def execute_train_unified_goodness(args):
    """
    Execute the train-unified-goodness command.

    Pipeline:
    1. Load ALL 327 benchmarks (same as test_all_benchmarks.py)
    2. Generate contrastive pairs from ALL selected benchmarks (pooled)
    3. Collect activations for all pairs
    4. Train single unified steering vector from pooled data
    5. Evaluate vector across ALL benchmarks (pooled evaluation)
    """
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.models.inference_config import get_generate_kwargs

    pipeline_start = time.time() if args.timing else None

    print("\n" + "=" * 70)
    print("🎯 UNIFIED GOODNESS VECTOR TRAINING")
    print("=" * 70)
    print("Training a single steering vector that improves performance")
    print("across ALL benchmarks simultaneously.")
    print("=" * 70 + "\n")

    # =========================================================================
    # Step 1: Select benchmarks - use ALL 327 benchmarks by default
    # =========================================================================
    print("📋 Step 1/5: Selecting benchmarks...")

    # Load ALL benchmarks from parameter files (same as test_all_benchmarks.py)
    all_benchmark_names, broken_benchmarks = load_all_benchmarks()
    print(f"   ✓ Loaded {len(all_benchmark_names)} total benchmarks")
    if broken_benchmarks:
        print(f"   ✓ Skipping {len(broken_benchmarks)} broken benchmarks")

    # Determine benchmarks from --task argument
    if args.task:
        # Parse comma-separated benchmarks
        task_benchmarks = [b.strip() for b in args.task.split(",")]
        selected_benchmark_names = [name for name in task_benchmarks if name in all_benchmark_names]
        unknown = [name for name in task_benchmarks if name not in all_benchmark_names]
        for name in unknown:
            print(f"   ⚠️  Unknown benchmark: {name}, skipping")
        print(f"   ✓ Using specified benchmarks: {', '.join(selected_benchmark_names)}")
    else:
        # Use ALL benchmarks by default
        selected_benchmark_names = all_benchmark_names.copy()

    # Apply exclusions
    if args.exclude_benchmarks:
        for name in args.exclude_benchmarks:
            if name in selected_benchmark_names:
                selected_benchmark_names.remove(name)
                if args.verbose:
                    print(f"   Excluded: {name}")

    # Apply max limit
    if args.max_benchmarks and len(selected_benchmark_names) > args.max_benchmarks:
        selected_benchmark_names = selected_benchmark_names[:args.max_benchmarks]

    print(f"   ✓ Selected {len(selected_benchmark_names)} benchmarks for training")
    if args.verbose:
        for name in selected_benchmark_names[:20]:
            print(f"      • {name}")
        if len(selected_benchmark_names) > 20:
            print(f"      ... and {len(selected_benchmark_names) - 20} more")

    # =========================================================================
    # Step 2: Load model
    # =========================================================================
    print(f"\n🤖 Step 2/5: Loading model '{args.model}'...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Model loaded with {model.num_layers} layers")
    print(f"   ✓ Hidden size: {model.hidden_size}")

    # Determine layer(s) to use
    if args.layer is not None:
        layers = [str(args.layer)]
    elif args.layers:
        # Parse layer specification
        layers = []
        for part in args.layers.replace(" ", "").split(","):
            if "-" in part or ".." in part:
                a, b = part.replace("..", "-").split("-")
                layers.extend(str(i) for i in range(int(a), int(b) + 1))
            else:
                layers.append(part)
    else:
        # Use middle layer by default
        middle_layer = model.num_layers // 2
        layers = [str(middle_layer)]
        print(f"   Using middle layer: {middle_layer}")

    print(f"   ✓ Target layers: {layers}")

    # =========================================================================
    # Step 3: Collect contrastive pairs from ALL benchmarks (POOLED)
    # =========================================================================
    print(f"\n📊 Step 3/5: Collecting contrastive pairs from all benchmarks...")

    # Set random seed for reproducibility
    random.seed(args.seed)

    loader = LMEvalDataLoader()
    all_train_pairs = []
    all_eval_pairs = []
    benchmark_pair_counts = {}
    failed_benchmarks = []

    for idx, bench_name in enumerate(selected_benchmark_names):
        task_name = bench_name  # Task name is the benchmark name
        print(f"   [{idx+1}/{len(selected_benchmark_names)}] Loading {bench_name}...", end=" ", flush=True)

        try:
            task_obj = loader.load_lm_eval_task(task_name)

            # Handle group tasks (dict of subtasks)
            if isinstance(task_obj, dict):
                bench_pairs = []
                for subname, subtask in task_obj.items():
                    try:
                        subtask_pairs = lm_build_contrastive_pairs(
                            task_name=subname,
                            lm_eval_task=subtask,
                            limit=None,  # Load all, we'll cap later
                        )
                        bench_pairs.extend(subtask_pairs)
                    except Exception as e:
                        if args.verbose:
                            print(f"\n      ⚠️  Subtask {subname} failed: {e}")
            else:
                # Load all pairs
                bench_pairs = lm_build_contrastive_pairs(
                    task_name=task_name,
                    lm_eval_task=task_obj,
                    limit=None,  # Load all, we'll cap later
                )

            # Apply cap with random sampling if needed
            if bench_pairs and args.cap_pairs_per_benchmark and len(bench_pairs) > args.cap_pairs_per_benchmark:
                original_count = len(bench_pairs)
                bench_pairs = random.sample(bench_pairs, args.cap_pairs_per_benchmark)
                if args.verbose:
                    print(f"(capped {original_count} -> {len(bench_pairs)}) ", end="")

            if bench_pairs:
                # Add source_benchmark to each pair's metadata using replace()
                from dataclasses import replace
                tagged_pairs = []
                for pair in bench_pairs:
                    existing_meta = pair.metadata or {}
                    new_meta = {**existing_meta, 'source_benchmark': bench_name}
                    tagged_pairs.append(replace(pair, metadata=new_meta))
                bench_pairs = tagged_pairs

                # Split into train/eval
                n_train = int(len(bench_pairs) * args.train_ratio)
                train_pairs = bench_pairs[:n_train]
                eval_pairs = bench_pairs[n_train:]

                all_train_pairs.extend(train_pairs)
                all_eval_pairs.extend(eval_pairs)
                benchmark_pair_counts[bench_name] = {
                    'train': len(train_pairs),
                    'eval': len(eval_pairs)
                }
                print(f"✓ {len(train_pairs)} train, {len(eval_pairs)} eval")
            else:
                print("⚠️  No pairs generated")
                failed_benchmarks.append(bench_name)

        except Exception as e:
            print(f"❌ Failed: {e}")
            failed_benchmarks.append(bench_name)
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\n   📈 POOLED DATA SUMMARY:")
    print(f"      Total training pairs: {len(all_train_pairs)}")
    print(f"      Total evaluation pairs: {len(all_eval_pairs)}")
    print(f"      Successful benchmarks: {len(benchmark_pair_counts)}")
    print(f"      Failed benchmarks: {len(failed_benchmarks)}")

    if not all_train_pairs:
        print("\n❌ No training pairs collected. Cannot proceed.")
        sys.exit(1)

    # Create ContrastivePairSet
    train_pair_set = ContrastivePairSet(
        name="unified_goodness_train",
        pairs=all_train_pairs,
        task_type="pooled_multi_benchmark"
    )

    # Save pairs if requested
    if args.save_pairs:
        print(f"\n   💾 Saving pooled pairs to {args.save_pairs}...")
        pairs_data = {
            'train_count': len(all_train_pairs),
            'eval_count': len(all_eval_pairs),
            'benchmark_counts': benchmark_pair_counts,
            'failed_benchmarks': failed_benchmarks,
        }
        with open(args.save_pairs, 'w') as f:
            json.dump(pairs_data, f, indent=2)

    # =========================================================================
    # Step 4: Collect activations and train steering vector
    # =========================================================================
    print(f"\n🧠 Step 4/5: Collecting activations and training vector...")

    # Map aggregation strategy
    aggregation_map = {
        'average': ActivationAggregationStrategy.MEAN_POOLING,
        'final': ActivationAggregationStrategy.LAST_TOKEN,
        'first': ActivationAggregationStrategy.FIRST_TOKEN,
        'max': ActivationAggregationStrategy.MAX_POOLING,
        'continuation': ActivationAggregationStrategy.CONTINUATION_TOKEN,
    }
    aggregation_strategy = aggregation_map.get(
        args.token_aggregation,
        ActivationAggregationStrategy.CONTINUATION_TOKEN
    )

    # Map prompt strategy
    prompt_strategy_map = {
        'chat_template': PromptConstructionStrategy.CHAT_TEMPLATE,
        'direct_completion': PromptConstructionStrategy.DIRECT_COMPLETION,
        'instruction_following': PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
        'multiple_choice': PromptConstructionStrategy.MULTIPLE_CHOICE,
        'role_playing': PromptConstructionStrategy.ROLE_PLAYING,
    }
    prompt_strategy = prompt_strategy_map.get(
        args.prompt_strategy,
        PromptConstructionStrategy.CHAT_TEMPLATE
    )

    collector = ActivationCollector(model=model, store_device="cpu")

    # Collect activations for all training pairs
    positive_activations = {layer: [] for layer in layers}
    negative_activations = {layer: [] for layer in layers}

    print(f"   Collecting activations for {len(all_train_pairs)} pairs...")
    for i, pair in enumerate(all_train_pairs):
        if i % 50 == 0:
            print(f"      Processing pair {i + 1}/{len(all_train_pairs)}...", end='\r', flush=True)

        try:
            updated_pair = collector.collect_for_pair(
                pair,
                layers=layers,
                aggregation=aggregation_strategy,
                return_full_sequence=False,
                normalize_layers=False,
                prompt_strategy=prompt_strategy
            )

            for layer in layers:
                if updated_pair.positive_response.layers_activations and layer in updated_pair.positive_response.layers_activations:
                    act = updated_pair.positive_response.layers_activations[layer]
                    if act is not None:
                        positive_activations[layer].append(act.cpu())

                if updated_pair.negative_response.layers_activations and layer in updated_pair.negative_response.layers_activations:
                    act = updated_pair.negative_response.layers_activations[layer]
                    if act is not None:
                        negative_activations[layer].append(act.cpu())

        except Exception as e:
            if args.verbose:
                print(f"\n      ⚠️  Pair {i} failed: {e}")

    print(f"\n   ✓ Collected activations from {len(positive_activations[layers[0]])} pairs")

    # Train steering vector using CAA
    print(f"\n   🎯 Training unified steering vector using {args.method.upper()}...")

    steering_method = CAAMethod(normalize=args.normalize)
    steering_vectors = {}

    for layer in layers:
        pos_list = positive_activations[layer]
        neg_list = negative_activations[layer]

        if pos_list and neg_list:
            vector = steering_method.train_for_layer(pos_list, neg_list)
            steering_vectors[layer] = vector
            print(f"      Layer {layer}: vector shape {vector.shape}, norm {torch.norm(vector).item():.4f}")

    # Save the unified steering vector
    print(f"\n   💾 Saving unified vector to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Use first layer as the primary vector (for single-layer output)
    primary_layer = layers[0]
    # Serialize eval pairs for later use (e.g., by optimize-weights)
    serialized_eval_pairs = []
    for pair in all_eval_pairs:
        serialized_eval_pairs.append({
            'prompt': pair.prompt,
            'positive_response': pair.positive_response.model_response,
            'negative_response': pair.negative_response.model_response,
            'source_benchmark': pair.metadata.get('source_benchmark') if pair.metadata else None,
            'metadata': pair.metadata,
        })

    save_data = {
        'steering_vector': steering_vectors[primary_layer],
        'layer_index': int(primary_layer),
        'method': args.method,
        'model': args.model,
        'type': 'unified_goodness',
        'benchmarks_used': list(benchmark_pair_counts.keys()),
        'num_benchmarks': len(benchmark_pair_counts),
        'num_training_pairs': len(all_train_pairs),
        'num_eval_pairs': len(all_eval_pairs),
        'benchmark_pair_counts': benchmark_pair_counts,
        'normalize': args.normalize,
        'aggregation': args.token_aggregation,
        # All layer vectors if multiple
        'all_layer_vectors': {k: v for k, v in steering_vectors.items()},
        # Eval pairs for optimize-weights to use
        'eval_pairs': serialized_eval_pairs,
        # Legacy keys for backward compatibility
        'vector': steering_vectors[primary_layer],
        'layer': int(primary_layer),
    }
    torch.save(save_data, args.output)
    print(f"   ✓ Saved unified goodness vector")

    # =========================================================================
    # Step 5: Pooled evaluation across ALL benchmarks
    # =========================================================================
    if args.skip_evaluation:
        print("\n⏭️  Skipping evaluation (--skip-evaluation flag)")
    else:
        print(f"\n📈 Step 5/5: Pooled evaluation across all benchmarks...")

        # Parse steering scales
        steering_scales = [float(s) for s in args.evaluate_steering_scales.split(",")]
        print(f"   Testing steering scales: {steering_scales}")

        # Discover evaluators
        EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
        EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")

        evaluation_results = {
            'by_scale': {},
            'by_benchmark': {},
            'summary': {}
        }

        for scale in steering_scales:
            print(f"\n   📊 Evaluating with steering scale {scale}...")
            scale_results = []

            # Create steering plan for this scale
            from wisent.core.models.core.atoms import SteeringPlan
            from wisent.core.activations.core.atoms import RawActivationMap

            if scale > 0:
                raw_map: RawActivationMap = {
                    primary_layer: steering_vectors[primary_layer]
                }
                steering_plan = SteeringPlan.from_raw(
                    raw=raw_map,
                    scale=scale,
                    normalize=False  # Already normalized during training
                )
            else:
                steering_plan = None

            # Evaluate on each benchmark that has eval pairs
            # Group eval pairs by their source benchmark
            eval_pairs_by_benchmark = {}
            for pair in all_eval_pairs:
                bench_name = pair.metadata.get('source_benchmark') if pair.metadata else None
                if bench_name:
                    if bench_name not in eval_pairs_by_benchmark:
                        eval_pairs_by_benchmark[bench_name] = []
                    eval_pairs_by_benchmark[bench_name].append(pair)

            for bench_name, bench_eval_pairs in eval_pairs_by_benchmark.items():
                task_name = bench_name  # Task name is the benchmark name
                evaluator = EvaluatorRotator(
                    evaluator=None,
                    task_name=task_name,
                    autoload=False
                )

                correct = 0
                total = 0

                for pair in bench_eval_pairs:  # Use all eval pairs (already 20% of benchmark)
                    try:
                        question = pair.prompt
                        expected = pair.positive_response.model_response
                        messages = [{"role": "user", "content": question}]

                        # Generate with or without steering
                        if steering_plan:
                            response = model.generate(
                                [messages],
                                **get_generate_kwargs(),
                                use_steering=True,
                                steering_plan=steering_plan,
                            )[0]
                        else:
                            response = model.generate(
                                [messages],
                                **get_generate_kwargs(),
                            )[0]

                        # Evaluate
                        eval_result = evaluator.evaluate(
                            response=response,
                            expected=expected,
                            model=model,
                            question=question,
                            task_name=task_name,
                        )

                        if eval_result.ground_truth == "TRUTHFUL":
                            correct += 1
                        total += 1

                    except Exception as e:
                        if args.verbose:
                            print(f"      ⚠️  Eval failed for {bench_name}: {e}")

                if total > 0:
                    accuracy = correct / total
                    scale_results.append({
                        'benchmark': bench_name,
                        'correct': correct,
                        'total': total,
                        'accuracy': accuracy
                    })

                    if bench_name not in evaluation_results['by_benchmark']:
                        evaluation_results['by_benchmark'][bench_name] = {}
                    evaluation_results['by_benchmark'][bench_name][scale] = accuracy

            # Compute aggregate metrics for this scale
            if scale_results:
                avg_accuracy = sum(r['accuracy'] for r in scale_results) / len(scale_results)
                total_correct = sum(r['correct'] for r in scale_results)
                total_samples = sum(r['total'] for r in scale_results)
                pooled_accuracy = total_correct / total_samples if total_samples > 0 else 0

                evaluation_results['by_scale'][scale] = {
                    'per_benchmark': scale_results,
                    'avg_accuracy': avg_accuracy,
                    'pooled_accuracy': pooled_accuracy,
                    'total_correct': total_correct,
                    'total_samples': total_samples
                }

                print(f"      Scale {scale}: Avg accuracy = {avg_accuracy:.4f}, Pooled = {pooled_accuracy:.4f}")

        # Find optimal scale
        if evaluation_results['by_scale']:
            best_scale = max(
                evaluation_results['by_scale'].keys(),
                key=lambda s: evaluation_results['by_scale'][s]['pooled_accuracy']
            )
            best_result = evaluation_results['by_scale'][best_scale]
            evaluation_results['summary'] = {
                'best_scale': best_scale,
                'best_pooled_accuracy': best_result['pooled_accuracy'],
                'best_avg_accuracy': best_result['avg_accuracy'],
                'baseline_pooled_accuracy': evaluation_results['by_scale'].get(0.0, {}).get('pooled_accuracy', 0),
                'improvement': best_result['pooled_accuracy'] - evaluation_results['by_scale'].get(0.0, {}).get('pooled_accuracy', 0)
            }

            print(f"\n   🏆 BEST SCALE: {best_scale}")
            print(f"      Pooled accuracy: {best_result['pooled_accuracy']:.4f}")
            print(f"      Avg benchmark accuracy: {best_result['avg_accuracy']:.4f}")
            if 0.0 in evaluation_results['by_scale']:
                baseline = evaluation_results['by_scale'][0.0]['pooled_accuracy']
                improvement = best_result['pooled_accuracy'] - baseline
                print(f"      Improvement over baseline: {improvement:+.4f}")

        # Save evaluation report
        if args.save_report:
            print(f"\n   💾 Saving evaluation report to {args.save_report}...")
            report = {
                'model': args.model,
                'output_vector': args.output,
                'benchmarks': list(benchmark_pair_counts.keys()),
                'failed_benchmarks': failed_benchmarks,
                'training_pairs': len(all_train_pairs),
                'eval_pairs': len(all_eval_pairs),
                'layers': layers,
                'evaluation': evaluation_results,
            }
            with open(args.save_report, 'w') as f:
                json.dump(report, f, indent=2)

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ UNIFIED GOODNESS VECTOR TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Output vector: {args.output}")
    print(f"   Benchmarks used: {len(benchmark_pair_counts)}")
    print(f"   Training pairs (pooled): {len(all_train_pairs)}")
    print(f"   Layers: {layers}")

    if args.timing and pipeline_start:
        total_time = time.time() - pipeline_start
        print(f"   ⏱️  Total time: {total_time:.2f}s")

    print("=" * 70 + "\n")

    return {
        'output': args.output,
        'benchmarks': list(benchmark_pair_counts.keys()),
        'training_pairs': len(all_train_pairs),
        'layers': layers,
        'evaluation': evaluation_results if not args.skip_evaluation else None
    }
