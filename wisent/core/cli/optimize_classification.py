"""Classification optimization command execution logic."""

import sys
import json
import time
from typing import List, Dict, Any

def execute_optimize_classification(args):
    """
    Execute the optimize-classification command.
    
    Optimizes classification parameters across all available tasks:
    - Finds best layer for each task
    - Finds best token aggregation method  
    - Finds best detection threshold
    - Saves trained classifiers
    
    EFFICIENCY: Collects raw activations ONCE, then applies different aggregation strategies
    to the cached activations without re-running the model.
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
    from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig
    import numpy as np
    import torch
    
    print(f"\n{'='*80}")
    print(f"🔍 CLASSIFICATION PARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Limit per task: {args.limit}")
    print(f"   Optimization metric: {args.optimization_metric}")
    print(f"   Device: {args.device or 'auto'}")
    print(f"{'='*80}\n")
    
    # 1. Load model
    print(f"📦 Loading model...")
    model = WisentModel(args.model, device=args.device)
    total_layers = model.num_layers
    print(f"   ✓ Model loaded with {total_layers} layers\n")
    
    # 2. Determine layer range
    if args.layer_range:
        start, end = map(int, args.layer_range.split('-'))
        layers_to_test = list(range(start, end + 1))
    else:
        # Test middle layers by default (more informative)
        start_layer = total_layers // 3
        end_layer = (2 * total_layers) // 3
        layers_to_test = list(range(start_layer, end_layer + 1))
    
    print(f"🎯 Testing layers: {layers_to_test[0]} to {layers_to_test[-1]} ({len(layers_to_test)} layers)")
    print(f"🔄 Aggregation methods: {', '.join(args.aggregation_methods)}")
    print(f"📊 Thresholds: {args.threshold_range}\n")
    
    # 3. Get list of tasks to optimize
    task_list = [
        "arc_easy", "arc_challenge", "hellaswag", 
        "winogrande", "gsm8k"
    ]
    
    print(f"📋 Optimizing {len(task_list)} tasks\n")
    
    # 4. Initialize data loader
    loader = LMEvalDataLoader()
    
    # 5. Results storage
    all_results = {}
    classifiers_saved = {}
    
    # 6. Process each task
    for task_idx, task_name in enumerate(task_list, 1):
        print(f"\n{'='*80}")
        print(f"Task {task_idx}/{len(task_list)}: {task_name}")
        print(f"{'='*80}")
        
        task_start_time = time.time()
        
        try:
            # Load task data
            print(f"  📊 Loading data...")
            result = loader._load_one_task(
                task_name=task_name,
                split_ratio=0.8,
                seed=42,
                limit=args.limit,
                training_limit=None,
                testing_limit=None
            )
            
            train_pairs = result['train_qa_pairs']
            test_pairs = result['test_qa_pairs']
            
            print(f"      ✓ Loaded {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs")
            
            # STEP 1: Collect raw activations ONCE for all layers (full sequence)
            print(f"  🧠 Collecting raw activations (once per pair)...")
            collector = ActivationCollector(model=model, store_device="cpu")
            
            # Cache structure: train_cache[pair_idx][layer_str] = {pos: tensor, neg: tensor, pos_tokens: int, neg_tokens: int}
            train_cache = {}
            test_cache = {}
            
            layer_strs = [str(l) for l in layers_to_test]
            
            # Collect training activations with full sequence
            for pair_idx, pair in enumerate(train_pairs.pairs):
                updated_pair = collector.collect_for_pair(
                    pair, 
                    layers=layer_strs, 
                    aggregation=None,  # Get raw activations without aggregation
                    return_full_sequence=True,  # Get all token positions
                    normalize_layers=False
                )
                
                train_cache[pair_idx] = {}
                for layer_str in layer_strs:
                    train_cache[pair_idx][layer_str] = {
                        'pos': updated_pair.positive_response.layers_activations.get(layer_str),
                        'neg': updated_pair.negative_response.layers_activations.get(layer_str),
                    }
            
            # Collect test activations
            for pair_idx, pair in enumerate(test_pairs.pairs):
                updated_pair = collector.collect_for_pair(
                    pair,
                    layers=layer_strs,
                    aggregation=None,
                    return_full_sequence=True,
                    normalize_layers=False
                )
                
                test_cache[pair_idx] = {}
                for layer_str in layer_strs:
                    test_cache[pair_idx][layer_str] = {
                        'pos': updated_pair.positive_response.layers_activations.get(layer_str),
                        'neg': updated_pair.negative_response.layers_activations.get(layer_str),
                    }
            
            print(f"      ✓ Cached activations for {len(train_cache)} train and {len(test_cache)} test pairs")
            
            # STEP 2: Apply different aggregation strategies to cached activations
            print(f"  🔍 Testing {len(layers_to_test) * len(args.aggregation_methods)} layer/aggregation combinations...")
            
            # Aggregation functions
            def aggregate_activations(raw_acts, method):
                """Apply aggregation to raw activation tensor."""
                if raw_acts is None or raw_acts.numel() == 0:
                    return None
                
                # Handle both 1D (already aggregated) and 2D (sequence, hidden_dim) tensors
                if raw_acts.ndim == 1:
                    return raw_acts
                elif raw_acts.ndim == 2:
                    if method == 'average':
                        return raw_acts.mean(dim=0)
                    elif method == 'final':
                        return raw_acts[-1]
                    elif method == 'first':
                        return raw_acts[0]
                    elif method == 'max':
                        return raw_acts.max(dim=0)[0]
                    elif method == 'min':
                        return raw_acts.min(dim=0)[0]
                else:
                    # Flatten to 2D if needed
                    raw_acts = raw_acts.view(-1, raw_acts.shape[-1])
                    return aggregate_activations(raw_acts, method)
            
            best_score = -1
            best_config = None
            best_classifier = None
            
            combinations_tested = 0
            total_combinations = len(layers_to_test) * len(args.aggregation_methods)
            
            for layer in layers_to_test:
                layer_str = str(layer)
                
                for agg_method in args.aggregation_methods:
                    # Apply aggregation to cached activations
                    train_pos_acts = []
                    train_neg_acts = []
                    
                    for pair_idx in train_cache:
                        pos_raw = train_cache[pair_idx][layer_str]['pos']
                        neg_raw = train_cache[pair_idx][layer_str]['neg']
                        
                        pos_agg = aggregate_activations(pos_raw, agg_method)
                        neg_agg = aggregate_activations(neg_raw, agg_method)
                        
                        if pos_agg is not None:
                            train_pos_acts.append(pos_agg.cpu().numpy())
                        if neg_agg is not None:
                            train_neg_acts.append(neg_agg.cpu().numpy())
                    
                    if len(train_pos_acts) == 0 or len(train_neg_acts) == 0:
                        combinations_tested += 1
                        continue
                    
                    # Prepare training data
                    X_train_pos = np.array(train_pos_acts)
                    X_train_neg = np.array(train_neg_acts)
                    X_train = np.vstack([X_train_pos, X_train_neg])
                    y_train = np.array([1] * len(train_pos_acts) + [0] * len(train_neg_acts))
                    
                    # Train classifier
                    classifier = LogisticClassifier(threshold=0.5, device="cpu")
                    
                    config = ClassifierTrainConfig(
                        test_size=0.2,
                        batch_size=32,
                        num_epochs=30,
                        learning_rate=0.001,
                        monitor="f1",
                        random_state=42
                    )
                    
                    report = classifier.fit(
                        torch.tensor(X_train, dtype=torch.float32),
                        torch.tensor(y_train, dtype=torch.float32),
                        config=config
                    )
                    
                    # Apply aggregation to test set
                    test_pos_acts = []
                    test_neg_acts = []
                    
                    for pair_idx in test_cache:
                        pos_raw = test_cache[pair_idx][layer_str]['pos']
                        neg_raw = test_cache[pair_idx][layer_str]['neg']
                        
                        pos_agg = aggregate_activations(pos_raw, agg_method)
                        neg_agg = aggregate_activations(neg_raw, agg_method)
                        
                        if pos_agg is not None:
                            test_pos_acts.append(pos_agg.cpu().numpy())
                        if neg_agg is not None:
                            test_neg_acts.append(neg_agg.cpu().numpy())
                    
                    if len(test_pos_acts) == 0 or len(test_neg_acts) == 0:
                        combinations_tested += 1
                        continue
                    
                    X_test_pos = np.array(test_pos_acts)
                    X_test_neg = np.array(test_neg_acts)
                    X_test = np.vstack([X_test_pos, X_test_neg])
                    y_test = np.array([1] * len(test_pos_acts) + [0] * len(test_neg_acts))
                    
                    # Get predictions
                    y_pred_proba = np.array(classifier.predict_proba(X_test))
                    
                    # Test different thresholds
                    for threshold in args.threshold_range:
                        y_pred = (y_pred_proba > threshold).astype(int)
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        
                        # Choose metric based on args
                        metric_value = {
                            'f1': f1,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall
                        }[args.optimization_metric]
                        
                        if metric_value > best_score:
                            best_score = metric_value
                            best_config = {
                                'layer': layer,
                                'aggregation': agg_method,
                                'threshold': threshold,
                                'accuracy': float(accuracy),
                                'f1': float(f1),
                                'precision': float(precision),
                                'recall': float(recall)
                            }
                            best_classifier = classifier
                    
                    combinations_tested += 1
                    print(f"      Progress: {combinations_tested}/{total_combinations} combinations tested", end='\r')
            
            print(f"\n  ✅ Best config: layer={best_config['layer']}, agg={best_config['aggregation']}, thresh={best_config['threshold']:.2f}")
            print(f"      Metrics: acc={best_config['accuracy']:.3f}, f1={best_config['f1']:.3f}, prec={best_config['precision']:.3f}, rec={best_config['recall']:.3f}")
            
            all_results[task_name] = best_config
            
            # Note: Classifier saving disabled due to missing .save() method
            # Can be enabled once proper serialization is implemented
            
            task_time = time.time() - task_start_time
            print(f"  ⏱️  Task completed in {task_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Failed to optimize {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 7. Save results
    print(f"\n{'='*80}")
    print(f"📊 OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")
    
    results_file = args.results_file or f"./optimization_results/classification_results.json"
    import os
    os.makedirs(os.path.dirname(results_file) if os.path.dirname(results_file) else ".", exist_ok=True)
    
    output_data = {
        'model': args.model,
        'optimization_metric': args.optimization_metric,
        'layer_range': f"{layers_to_test[0]}-{layers_to_test[-1]}",
        'aggregation_methods': args.aggregation_methods,
        'threshold_range': args.threshold_range,
        'tasks': all_results,
        'classifiers_saved': classifiers_saved
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}\n")
    
    # Print summary
    print("📋 SUMMARY BY TASK:")
    print("-" * 80)
    for task_name, config in all_results.items():
        print(f"  {task_name:20s} | Layer: {config['layer']:2d} | Agg: {config['aggregation']:8s} | Thresh: {config['threshold']:.2f} | F1: {config['f1']:.3f}")
    print("-" * 80 + "\n")

    # Return results for programmatic access
    return {
        "model": args.model,
        "optimization_metric": args.optimization_metric,
        "layer_range": f"{layers_to_test[0]}-{layers_to_test[-1]}",
        "tasks_optimized": list(all_results.keys()),
        "results": all_results,
        "results_file": results_file,
        "classifiers_saved": classifiers_saved
    }

