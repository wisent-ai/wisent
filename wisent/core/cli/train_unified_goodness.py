"""Train unified goodness vector from pooled multi-benchmark data."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch


def get_checkpoint_dir(output_path: str) -> Path:
    """Get checkpoint directory based on output path."""
    output_dir = Path(output_path).parent
    checkpoint_dir = output_dir / ".unified_goodness_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(checkpoint_dir: Path, name: str, data: Any):
    """Save checkpoint data."""
    checkpoint_path = checkpoint_dir / f"{name}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 Checkpoint saved: {name}")


def load_checkpoint(checkpoint_dir: Path, name: str) -> Optional[Any]:
    """Load checkpoint data if it exists."""
    checkpoint_path = checkpoint_dir / f"{name}.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   📂 Checkpoint loaded: {name}")
        return data
    return None


def clear_checkpoints(checkpoint_dir: Path):
    """Clear all checkpoints after successful completion."""
    import shutil
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"   🧹 Checkpoints cleared")


def load_all_benchmarks():
    """
    Load ALL benchmarks from the central registry.
    
    Returns:
        Tuple of (filtered_benchmarks, broken_benchmarks)
    """
    from wisent.core.benchmark_registry import load_all_benchmarks as _load_all_benchmarks
    return _load_all_benchmarks()


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
    # Expand task if it's a skill or risk name
    from wisent.core.task_selector import expand_task_if_skill_or_risk
    if args.task:
        args.task = expand_task_if_skill_or_risk(args.task)
    
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
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

    # Setup checkpointing
    checkpoint_dir = get_checkpoint_dir(args.output)

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
        # Use ALL layers by default
        layers = [str(i) for i in range(model.num_layers)]
        print(f"   Using ALL layers: 0 to {model.num_layers - 1}")

    print(f"   ✓ Target layers: {layers}")

    # =========================================================================
    # Step 3: Collect contrastive pairs from ALL benchmarks (POOLED)
    # =========================================================================
    print(f"\n📊 Step 3/5: Collecting contrastive pairs from all benchmarks...")

    # Try to load pairs from checkpoint
    pairs_checkpoint = load_checkpoint(checkpoint_dir, "pairs_data")
    
    if pairs_checkpoint is not None:
        all_train_pairs = pairs_checkpoint['all_train_pairs']
        all_eval_pairs = pairs_checkpoint['all_eval_pairs']
        benchmark_pair_counts = pairs_checkpoint['benchmark_pair_counts']
        failed_benchmarks = pairs_checkpoint['failed_benchmarks']
        print(f"   ✓ Loaded {len(all_train_pairs)} train pairs, {len(all_eval_pairs)} eval pairs from checkpoint")
    else:
        # Set random seed for reproducibility
        random.seed(args.seed)

        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import HF_EXTRACTORS
        
        loader = LMEvalDataLoader()
        all_train_pairs = []
        all_eval_pairs = []
        benchmark_pair_counts = {}
        failed_benchmarks = []

        for idx, bench_name in enumerate(selected_benchmark_names):
            task_name = bench_name  # Task name is the benchmark name
            print(f"   [{idx+1}/{len(selected_benchmark_names)}] Loading {bench_name}...", end=" ", flush=True)

            try:
                # Check if task is in HuggingFace manifest (doesn't need lm-eval loading)
                task_name_lower = task_name.lower()
                is_hf_task = task_name_lower in {k.lower() for k in HF_EXTRACTORS.keys()}
                
                if is_hf_task:
                    # HuggingFace task - skip lm-eval loading, go directly to extractor
                    bench_pairs = lm_build_contrastive_pairs(
                        task_name=task_name,
                        lm_eval_task=None,  # HF extractors don't need lm_eval_task
                        limit=None,  # Load all, we'll cap later
                    )
                else:
                    # lm-eval task - load via LMEvalDataLoader
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

        # Save pairs checkpoint after collection
        save_checkpoint(checkpoint_dir, "pairs_data", {
            'all_train_pairs': all_train_pairs,
            'all_eval_pairs': all_eval_pairs,
            'benchmark_pair_counts': benchmark_pair_counts,
            'failed_benchmarks': failed_benchmarks,
        })

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
        'average': ExtractionStrategy.CHAT_MEAN,
        'final': ExtractionStrategy.CHAT_LAST,
        'first': ExtractionStrategy.CHAT_FIRST,
        'max': ExtractionStrategy.CHAT_MAX_NORM,
        'continuation': ExtractionStrategy.CHAT_FIRST,  # First answer token
    }
    aggregation_strategy = aggregation_map.get(
        args.token_aggregation,
        ExtractionStrategy.CHAT_LAST
    )

    # Map prompt strategy
    prompt_strategy_map = {
        'chat_template': ExtractionStrategy.CHAT_LAST,
        'direct_completion': ExtractionStrategy.CHAT_LAST,
        'instruction_following': ExtractionStrategy.CHAT_LAST,
        'multiple_choice': ExtractionStrategy.MC_BALANCED,
        'role_playing': ExtractionStrategy.ROLE_PLAY,
    }
    prompt_strategy = prompt_strategy_map.get(
        args.prompt_strategy,
        ExtractionStrategy.CHAT_LAST
    )

    # Try to load activations from checkpoint
    activations_checkpoint = load_checkpoint(checkpoint_dir, "activations_data")
    
    if activations_checkpoint is not None:
        positive_activations = activations_checkpoint['positive_activations']
        negative_activations = activations_checkpoint['negative_activations']
        print(f"   ✓ Loaded activations from checkpoint ({len(positive_activations[layers[0]])} pairs)")
    else:
        collector = ActivationCollector(model=model)

        # Collect activations for all training pairs using batched processing
        positive_activations = {layer: [] for layer in layers}
        negative_activations = {layer: [] for layer in layers}

        print(f"   Collecting activations for {len(all_train_pairs)} pairs (batched)...")
        
        # Build full texts for positive and negative responses
        tok = model.tokenizer
        positive_texts = []
        negative_texts = []
        
        for pair in all_train_pairs:
            # Build chat-formatted texts
            try:
                pos_text = tok.apply_chat_template(
                    [{"role": "user", "content": pair.prompt},
                     {"role": "assistant", "content": pair.positive_response.model_response}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                neg_text = tok.apply_chat_template(
                    [{"role": "user", "content": pair.prompt},
                     {"role": "assistant", "content": pair.negative_response.model_response}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback for models without chat templates
                pos_text = f"{pair.prompt} {pair.positive_response.model_response}"
                neg_text = f"{pair.prompt} {pair.negative_response.model_response}"
            
            positive_texts.append(pos_text)
            negative_texts.append(neg_text)
        
        # Collect positive activations in batches
        print(f"   Collecting positive response activations...")
        pos_results = collector.collect_batched(
            positive_texts,
            layers=layers,
            aggregation=aggregation_strategy,
            batch_size=1,
            show_progress=True,
        )
        
        # Collect negative activations in batches
        print(f"   Collecting negative response activations...")
        neg_results = collector.collect_batched(
            negative_texts,
            layers=layers,
            aggregation=aggregation_strategy,
            batch_size=1,
            show_progress=True,
        )
        
        # Organize results by layer
        for i, (pos_act, neg_act) in enumerate(zip(pos_results, neg_results)):
            for layer in layers:
                if layer in pos_act:
                    positive_activations[layer].append(pos_act[layer])
                if layer in neg_act:
                    negative_activations[layer].append(neg_act[layer])

        print(f"   ✓ Collected activations from {len(positive_activations[layers[0]])} pairs")
        
        # Save activations checkpoint
        save_checkpoint(checkpoint_dir, "activations_data", {
            'positive_activations': positive_activations,
            'negative_activations': negative_activations,
        })

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
    
    # Clear checkpoints on successful completion
    clear_checkpoints(checkpoint_dir)
    
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
