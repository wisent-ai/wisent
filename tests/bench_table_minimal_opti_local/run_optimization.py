"""
Main optimization script for bench_table_minimal_opti.

Runs complete optimization over:
- Layers
- Aggregation methods
- Prompt strategies
- Classification thresholds
- Classifier architectures (logistic, mlp)

For each configuration tuple, runs Optuna HPO and evaluates on test set.
"""

from __future__ import annotations

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import time
import traceback
from typing import List, Dict, Any
from datetime import datetime

from wisent.core.activations.core.atoms import ActivationAggregationStrategy
from wisent.core.models.wisent_model import WisentModel

from tests.bench_table_minimal_opti_local.boolq import boolq_config
from tests.bench_table_minimal_opti_local.cb import cb_config
from tests.bench_table_minimal_opti_local.gsm8k import gsm8k_config
from tests.bench_table_minimal_opti_local.sst2 import sst2_config
from tests.bench_table_minimal_opti_local.utils.activations_extractor import extract_activations_for_config, load_pairs
from tests.bench_table_minimal_opti_local.utils.classifier_optimization import (
    optimize_classifier_config,
    evaluate_on_test_set,
)
from tests.bench_table_minimal_opti_local.utils.memory_utils import cleanup_gpu_memory


AVAILABLE_BENCHMARKS = ["boolq", "cb", "gsm8k", "sst2"]

# commenting some of lines from config for testing

# Configuration spaces
AGGREGATION_METHODS = [
    #ActivationAggregationStrategy.LAST_TOKEN,
    #ActivationAggregationStrategy.FIRST_TOKEN,
    #ActivationAggregationStrategy.MEAN_POOLING,
    ActivationAggregationStrategy.CHOICE_TOKEN,
    ActivationAggregationStrategy.MAX_POOLING,
]

PROMPT_STRATEGIES = [
    "multiple_choice",
    "direct_completion",
    #"role_playing",
    #"instruction_following",
]

#THRESHOLDS = [0.25, 0.5, 0.75]
THRESHOLDS = [0.5]

#CLASSIFIER_TYPES = ["logistic", "mlp"]
CLASSIFIER_TYPES = ["mlp"]

def run_baseline_evaluation(
    benchmark: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
) -> Dict[str, Any]:
    """
    Run baseline performance evaluation for a benchmark.

    Args:
        benchmark: Benchmark name (boolq, cb, gsm8k, sst2)
        model_name: HuggingFace model name

    Returns:
        Dictionary with baseline results
    """
    print("\n" + "=" * 80)
    print(f"BASELINE EVALUATION: {benchmark.upper()}")
    print("=" * 80)

    # Get benchmark-specific config and evaluator
    if benchmark == "boolq":
        config_cls = boolq_config.BoolQConfig
        evaluator_cls = boolq_config.BoolQEvaluator
    elif benchmark == "cb":
        config_cls = cb_config.CBConfig
        evaluator_cls = cb_config.CBEvaluator
    elif benchmark == "gsm8k":
        config_cls = gsm8k_config.GSM8KConfig
        evaluator_cls = gsm8k_config.GSM8KEvaluator
    elif benchmark == "sst2":
        config_cls = sst2_config.SST2Config
        evaluator_cls = sst2_config.SST2Evaluator
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Get configs from config class
    data_config = config_cls.get_data_config()
    baseline_config = config_cls.get_baseline_config()

    # Get parameters from configs
    num_test = baseline_config["num_test"]
    max_new_tokens = baseline_config["max_new_tokens"]
    test_doc = data_config["test_source"]

    # Create evaluator with config values
    evaluator = evaluator_cls(
        model_name=model_name,
        num_questions=num_test,
        max_new_tokens=max_new_tokens,
        preferred_doc=test_doc,
    )

    baseline_results = evaluator.evaluate(output_path=None)
    return baseline_results

def run_configuration_optimization(
    benchmark: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
) -> Dict[str, Any]:
    """
    Run configuration optimization for a benchmark.

    Optimizes over (layer, aggregation, prompt_strategy, threshold, classifier_type) tuples.

    Args:
        benchmark: Benchmark name (boolq, cb, gsm8k, sst2)
        model_name: HuggingFace model name

    Returns:
        Dictionary with optimization results
    """
    print("\n" + "=" * 80)
    print(f"CONFIGURATION OPTIMIZATION: {benchmark.upper()}")
    print("=" * 80)

    # Get benchmark-specific config
    if benchmark == "boolq":
        config_cls = boolq_config.BoolQConfig
    elif benchmark == "cb":
        config_cls = cb_config.CBConfig
    elif benchmark == "gsm8k":
        config_cls = gsm8k_config.GSM8KConfig
    elif benchmark == "sst2":
        config_cls = sst2_config.SST2Config
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Get configs from config class
    data_config = config_cls.get_data_config()
    opti_config = config_cls.get_optimization_config()

    # Get optimization parameters from config
    num_train = opti_config["num_train"]
    num_val = opti_config["num_val"]
    num_test = opti_config["num_test"]
    n_trials = opti_config["n_trials"]
    n_runs = opti_config["n_runs"]

    train_val_doc = data_config["train_val_source"]
    test_doc = data_config["test_source"]

    # Load pairs once
    print(f"\nLoading train+val pairs from {train_val_doc}...")
    train_val_pairs = load_pairs(benchmark, num_train + num_val, preferred_doc=train_val_doc)

    print(f"Loading test pairs from {test_doc}...")
    test_pairs = load_pairs(benchmark, num_test, preferred_doc=test_doc)

    # Get number of layers
    print(f"\nLoading model to get number of layers...")
    model = WisentModel(model_name=model_name)
    num_layers = model.num_layers
    del model
    cleanup_gpu_memory(sleep_time=10)

    all_results = []
    best_config = None
    best_accuracy = 0.0

    total_configs = (
        num_layers *
        len(AGGREGATION_METHODS) *
        len(PROMPT_STRATEGIES) *
        len(THRESHOLDS) *
        len(CLASSIFIER_TYPES)
    )

    print(f"\nTotal configurations to evaluate: {total_configs}")
    config_idx = 0

    # Loop over all configuration tuples
    for layer in range(1, num_layers + 1):
        for agg_method in AGGREGATION_METHODS:
            for prompt_strategy in PROMPT_STRATEGIES:
                # Extract activations once per (layer, agg, prompt) combination
                print(f"\nExtracting activations for layer={layer}, agg={agg_method.value}, prompt={prompt_strategy}...")

                try:
                    print(f"  Extracting train+val activations...")
                    train_val_pos, train_val_neg = extract_activations_for_config(
                        model_name=model_name,
                        benchmark=benchmark,
                        pairs=train_val_pairs,
                        layer=layer,
                        aggregation_method=agg_method,
                        prompt_strategy=prompt_strategy,
                    )

                    # Aggressive cleanup between extractions
                    cleanup_gpu_memory(sleep_time=10)

                    print(f"  Extracting test activations...")
                    test_pos, test_neg = extract_activations_for_config(
                        model_name=model_name,
                        benchmark=benchmark,
                        pairs=test_pairs,
                        layer=layer,
                        aggregation_method=agg_method,
                        prompt_strategy=prompt_strategy,
                    )

                    # Aggressive cleanup after activation extraction
                    cleanup_gpu_memory(sleep_time=10)

                    # Convert to lists of tensors
                    train_val_pos_list = [train_val_pos[i] for i in range(len(train_val_pos))]
                    train_val_neg_list = [train_val_neg[i] for i in range(len(train_val_neg))]
                    test_pos_list = [test_pos[i] for i in range(len(test_pos))]
                    test_neg_list = [test_neg[i] for i in range(len(test_neg))]

                    # Now loop over threshold and classifier_type
                    for threshold in THRESHOLDS:
                        for classifier_type in CLASSIFIER_TYPES:
                            config_idx += 1

                            print(f"\n[{config_idx}/{total_configs}] Configuration:")
                            print(f"  Layer: {layer}")
                            print(f"  Aggregation: {agg_method.value}")
                            print(f"  Prompt strategy: {prompt_strategy}")
                            print(f"  Threshold: {threshold}")
                            print(f"  Classifier: {classifier_type}")

                            try:
                                # Run Optuna optimization
                                print(f"  Running Optuna optimization...")
                                best_params, best_val_acc = optimize_classifier_config(
                                    train_val_pos=train_val_pos_list,
                                    train_val_neg=train_val_neg_list,
                                    num_train=num_train,
                                    classifier_type=classifier_type,
                                    threshold=threshold,
                                    n_trials=n_trials,
                                )

                                print(f"  Best val accuracy: {best_val_acc:.3f}")

                                # Evaluate on test set
                                print(f"  Evaluating on test set ({n_runs} runs)...")
                                mean_acc, std_acc, all_accs = evaluate_on_test_set(
                                    train_val_pos=train_val_pos_list,
                                    train_val_neg=train_val_neg_list,
                                    test_pos=test_pos_list,
                                    test_neg=test_neg_list,
                                    num_train=num_train,
                                    classifier_type=classifier_type,
                                    best_params=best_params,
                                    threshold=threshold,
                                    n_runs=n_runs,
                                )

                                print(f"  Test accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

                                # Store results
                                config_result = {
                                    "layer": layer,
                                    "aggregation": agg_method.value,
                                    "prompt_strategy": prompt_strategy,
                                    "threshold": threshold,
                                    "classifier_type": classifier_type,
                                    "best_hyperparams": best_params,
                                    "best_val_accuracy": best_val_acc,
                                    "mean_test_accuracy": mean_acc,
                                    "std_test_accuracy": std_acc,
                                    "all_test_accuracies": all_accs,
                                }

                                all_results.append(config_result)

                                # Track best configuration
                                if mean_acc > best_accuracy:
                                    best_accuracy = mean_acc
                                    best_config = config_result

                            except Exception as e:
                                print(f"  ERROR: {e}")
                                traceback.print_exc()

                except Exception as e:
                    print(f"  ERROR extracting activations: {e}")
                    traceback.print_exc()
                    # Skip all threshold/classifier combinations for this (layer, agg, prompt)
                    config_idx += len(THRESHOLDS) * len(CLASSIFIER_TYPES)
                    continue

    return {
        "all_configurations": all_results,
        "best_configuration": best_config,
        "best_accuracy": best_accuracy,
    }


def run_benchmark_optimization(
    benchmark: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run complete optimization for a single benchmark.

    Calls baseline evaluation and configuration optimization separately.

    Args:
        benchmark: Benchmark name (boolq, cb, gsm8k, sst2)
        model_name: HuggingFace model name
        output_dir: Directory to save results

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {benchmark.upper()}")
    print("=" * 80)

    """
    # Step 1: Run baseline evaluation
    baseline_results = run_baseline_evaluation(
        benchmark=benchmark,
        model_name=model_name,
    )
    """


    # Step 2: Run configuration optimization
    optimization_results = run_configuration_optimization(
        benchmark=benchmark,
        model_name=model_name,
    )

    # Extract summary metrics
    baseline_accuracy = baseline_results["summary"]["accuracy"]
    best_optimization_accuracy = optimization_results["best_accuracy"]

    # Create combined results
    combined_results = {
        "benchmark": benchmark,
        "baseline_accuracy": baseline_accuracy,
        "best_optimization_accuracy": best_optimization_accuracy,
        "baseline": baseline_results,
        "optimization": optimization_results,
    }

    # Save combined results
    benchmark_output_dir = os.path.join(output_dir, benchmark)
    os.makedirs(benchmark_output_dir, exist_ok=True)

    combined_path = os.path.join(benchmark_output_dir, f"{benchmark}_results.json")
    with open(combined_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nResults saved to: {combined_path}")

    return combined_results


def run_multiple_benchmarks(
    benchmarks: List[str],
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run optimization for multiple benchmarks.

    Args:
        benchmarks: List of benchmark names
        model_name: HuggingFace model name
        output_dir: Directory to save results

    Returns:
        Dictionary with results for all benchmarks
    """
    # Validate benchmarks
    invalid = [b for b in benchmarks if b not in AVAILABLE_BENCHMARKS]
    if invalid:
        raise ValueError(
            f"Invalid benchmark(s): {invalid}\n"
            f"Available benchmarks: {AVAILABLE_BENCHMARKS}"
        )

    print("\n" + "=" * 80)
    print("MINIMAL OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"\nBenchmarks: {', '.join(benchmarks)}")
    print(f"Model: {model_name}")
    print(f"\nOptimization space:")
    print(f"  Aggregations: {len(AGGREGATION_METHODS)}")
    print(f"  Prompt strategies: {len(PROMPT_STRATEGIES)}")
    print(f"  Thresholds: {len(THRESHOLDS)}")
    print(f"  Classifier types: {len(CLASSIFIER_TYPES)}")
    total_configs_per_layer = (
        len(AGGREGATION_METHODS) *
        len(PROMPT_STRATEGIES) *
        len(THRESHOLDS) *
        len(CLASSIFIER_TYPES)
    )
    print(f"  Total configs per layer: {total_configs_per_layer}")
    print(f"\nNote: Each benchmark uses its own optimization config")
    print()

    overall_start = time.time()
    all_results = {}

    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n{'#' * 80}")
        print(f"# BENCHMARK {i}/{len(benchmarks)}: {benchmark.upper()}")
        print(f"{'#' * 80}")

        try:
            result = run_benchmark_optimization(
                benchmark=benchmark,
                model_name=model_name,
                output_dir=output_dir,
            )
            all_results[benchmark] = result

        except Exception as e:
            print(f"\n❌ Error running {benchmark}: {e}")
            traceback.print_exc()
            all_results[benchmark] = {"error": str(e)}

    overall_elapsed = time.time() - overall_start

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for benchmark, result in all_results.items():
        if "error" in result:
            print(f"\n✗ {benchmark.upper()}: FAILED")
            print(f"  Error: {result['error']}")
        else:
            baseline_acc = result["baseline"]["summary"]["accuracy"]
            best_config = result["optimization"]["best_configuration"]
            best_acc = best_config["mean_test_accuracy"]

            print(f"\n✓ {benchmark.upper()}")
            print(f"  Baseline accuracy: {baseline_acc:.2%}")
            print(f"  Best accuracy: {best_acc:.2%}")
            print(f"  Improvement: {(best_acc - baseline_acc):.2%}")
            print(f"  Best config:")
            print(f"    Layer: {best_config['layer']}")
            print(f"    Aggregation: {best_config['aggregation']}")
            print(f"    Prompt strategy: {best_config['prompt_strategy']}")
            print(f"    Threshold: {best_config['threshold']}")
            print(f"    Classifier: {best_config['classifier_type']}")

    hours, remainder = divmod(overall_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"\nAll results saved to: {output_dir}")

    return all_results


def main():
    """Main entry point."""
    try:
        # Run all benchmarks
        run_multiple_benchmarks(["sst2"])
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
