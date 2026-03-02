"""Data collection and training for unified goodness."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from wisent.core import constants as _C


def collect_pairs_and_train(args, wisent_model, layers, checkpoint_dir, benchmarks, loader):
    """Collect pairs from benchmarks, train steering vector. Returns (vectors, train, eval, benchmarks)."""
    from wisent.core.cli.analysis.training.train_unified_goodness import (
        save_checkpoint, load_checkpoint,
    )

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
            json.dump(pairs_data, f, indent=_C.JSON_INDENT)

    # =========================================================================
    # Step 4: Collect activations and train steering vector
    # =========================================================================
    print(f"\n🧠 Step 4/5: Collecting activations and training vector...")

    # Use centralized default extraction strategy
    aggregation_strategy = ExtractionStrategy.default()

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
            batch_size=_C.COMPARISON_DEFAULT_BATCH_SIZE,
            show_progress=True,
        )
        
        # Collect negative activations in batches
        print(f"   Collecting negative response activations...")
        neg_results = collector.collect_batched(
            negative_texts,
            layers=layers,
            aggregation=aggregation_strategy,
            batch_size=_C.COMPARISON_DEFAULT_BATCH_SIZE,
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
        'aggregation': ExtractionStrategy.default().name.lower(),
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


    return all_layer_vectors, train_pairs, eval_pairs, benchmarks_used
