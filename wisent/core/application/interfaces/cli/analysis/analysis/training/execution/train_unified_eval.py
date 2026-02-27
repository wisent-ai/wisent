"""Evaluation phase for unified goodness training."""
import json
import os
from pathlib import Path

from wisent.core.constants import JSON_INDENT, SEPARATOR_WIDTH_WIDE


def run_evaluation(args, wisent_model, all_layer_vectors, train_pairs, eval_pairs, benchmarks_used, checkpoint_dir):
    """Run Step 5: pooled evaluation and generate summary."""
    from wisent.core.cli.analysis.training.train_unified_goodness import (
        save_checkpoint, load_checkpoint, clear_checkpoints,
    )
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
                json.dump(report, f, indent=JSON_INDENT)

    # =========================================================================
    # Final summary
    # =========================================================================
    
    # Clear checkpoints on successful completion
    clear_checkpoints(checkpoint_dir)
    
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("✅ UNIFIED GOODNESS VECTOR TRAINING COMPLETE")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"   Output vector: {args.output}")
    print(f"   Benchmarks used: {len(benchmark_pair_counts)}")
    print(f"   Training pairs (pooled): {len(all_train_pairs)}")
    print(f"   Layers: {layers}")

    if args.timing and pipeline_start:
        total_time = time.time() - pipeline_start
        print(f"   ⏱️  Total time: {total_time:.2f}s")

    print("=" * SEPARATOR_WIDTH_WIDE + "\n")

    return {
        'output': args.output,
        'benchmarks': list(benchmark_pair_counts.keys()),
        'training_pairs': len(all_train_pairs),
        'layers': layers,
        'evaluation': evaluation_results if not args.skip_evaluation else None
    }
