"""
Evaluation metrics for comprehensive evaluation pipeline.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from wisent.core.constants import (
    BLEND_DEFAULT,
    DEFAULT_SCORE,
    EFFECTIVENESS_HIGH,
    EFFECTIVENESS_MODERATE,
    EFFECTIVENESS_SLIGHT,
    MATH_PERCENT_REL_TOL,
    SEPARATOR_WIDTH_STANDARD,
)
from wisent.extractors.lm_eval.mbpp import MBPPExtractor

# Import LMEvalHarnessGroundTruth for intelligent evaluation (newer approach used by CLI)
from wisent.core.lm_eval_harness_ground_truth import LMEvalHarnessGroundTruth
from wisent.core.tasks.base.task_interface import get_task
from wisent.core.tasks.file_task import FileTask
from wisent.core.errors import (
    EvaluationError,
    ExactMatchError,
    BigCodeEvaluationError,
    TaskNameRequiredError,
    InsufficientDataError,
)

# Use the standard evaluator system
from wisent.core.evaluators.rotator import EvaluatorRotator

logger = logging.getLogger(__name__)



def calculate_comprehensive_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics from evaluation results.

    Args:
        results: Complete evaluation results

    Returns:
        Dictionary with comprehensive metrics and analysis
    """
    comprehensive_metrics = {}

    if "test_results" in results:
        test_results = results["test_results"]

        # Extract key metrics
        base_benchmark_acc = test_results.get("base_benchmark_metrics", {}).get("accuracy", DEFAULT_SCORE)
        steered_benchmark_acc = test_results.get("steered_benchmark_metrics", {}).get("accuracy", DEFAULT_SCORE)
        base_probe_auc = test_results.get("base_probe_metrics", {}).get("auc", BLEND_DEFAULT)
        steered_probe_auc = test_results.get("steered_probe_metrics", {}).get("auc", BLEND_DEFAULT)

        # Calculate improvements
        benchmark_improvement = steered_benchmark_acc - base_benchmark_acc
        probe_improvement = steered_probe_auc - base_probe_auc

        comprehensive_metrics.update(
            {
                "base_benchmark_accuracy": base_benchmark_acc,
                "steered_benchmark_accuracy": steered_benchmark_acc,
                "benchmark_improvement": benchmark_improvement,
                "benchmark_improvement_percent": (benchmark_improvement / max(base_benchmark_acc, MATH_PERCENT_REL_TOL)) * 100,
                "base_probe_auc": base_probe_auc,
                "steered_probe_auc": steered_probe_auc,
                "probe_improvement": probe_improvement,
                "probe_improvement_percent": (probe_improvement / max(base_probe_auc, MATH_PERCENT_REL_TOL)) * 100,
                "overall_effectiveness": (benchmark_improvement + probe_improvement) / 2,
                "validation_score": test_results.get("validation_combined_score", DEFAULT_SCORE),
            }
        )

    # Add training statistics
    if "probe_training_results" in results:
        training_results = results["probe_training_results"]

        # Calculate training performance statistics
        all_training_aucs = []
        for layer_key, layer_results in training_results.items():
            for c_key, metrics in layer_results.items():
                if isinstance(metrics, dict) and "auc" in metrics:
                    all_training_aucs.append(metrics["auc"])

        if all_training_aucs:
            comprehensive_metrics.update(
                {
                    "training_probe_auc_mean": np.mean(all_training_aucs),
                    "training_probe_auc_std": np.std(all_training_aucs),
                    "training_probe_auc_max": np.max(all_training_aucs),
                    "training_probe_auc_min": np.min(all_training_aucs),
                }
            )

    # Add optimization statistics
    if "steering_optimization_results" in results:
        optimization_results = results["steering_optimization_results"]

        all_configs = optimization_results.get("all_configs", [])
        if all_configs:
            combined_scores = [config.get("combined_score", DEFAULT_SCORE) for config in all_configs]
            benchmark_scores = [config.get("benchmark_metrics", {}).get("accuracy", DEFAULT_SCORE) for config in all_configs]

            comprehensive_metrics.update(
                {
                    "optimization_configs_tested": len(all_configs),
                    "optimization_score_mean": np.mean(combined_scores),
                    "optimization_score_std": np.std(combined_scores),
                    "optimization_benchmark_mean": np.mean(benchmark_scores),
                    "optimization_benchmark_std": np.std(benchmark_scores),
                }
            )

    return comprehensive_metrics


def generate_performance_summary(comprehensive_metrics: Dict[str, Any]) -> str:
    """
    Generate a human-readable performance summary.

    Args:
        comprehensive_metrics: Comprehensive metrics dictionary

    Returns:
        String summary of performance
    """
    summary = []
    summary.append("=" * SEPARATOR_WIDTH_STANDARD)
    summary.append("COMPREHENSIVE EVALUATION PERFORMANCE SUMMARY")
    summary.append("=" * SEPARATOR_WIDTH_STANDARD)

    # Benchmark Performance
    if "base_benchmark_accuracy" in comprehensive_metrics:
        base_acc = comprehensive_metrics["base_benchmark_accuracy"]
        steered_acc = comprehensive_metrics["steered_benchmark_accuracy"]
        improvement = comprehensive_metrics["benchmark_improvement"]

        summary.append("\n📊 BENCHMARK PERFORMANCE:")
        summary.append(f"  Base Model Accuracy:    {base_acc:.3f} ({base_acc * 100:.1f}%)")
        summary.append(f"  Steered Model Accuracy: {steered_acc:.3f} ({steered_acc * 100:.1f}%)")
        summary.append(f"  Improvement:            {improvement:+.3f} ({improvement * 100:+.1f}%)")

    # Probe Performance
    if "base_probe_auc" in comprehensive_metrics:
        base_auc = comprehensive_metrics["base_probe_auc"]
        steered_auc = comprehensive_metrics["steered_probe_auc"]
        improvement = comprehensive_metrics["probe_improvement"]

        summary.append("\n🔍 PROBE PERFORMANCE:")
        summary.append(f"  Base Model Probe AUC:    {base_auc:.3f}")
        summary.append(f"  Steered Model Probe AUC: {steered_auc:.3f}")
        summary.append(f"  Improvement:             {improvement:+.3f}")

    # Training Statistics
    if "training_probe_auc_mean" in comprehensive_metrics:
        mean_auc = comprehensive_metrics["training_probe_auc_mean"]
        std_auc = comprehensive_metrics["training_probe_auc_std"]
        max_auc = comprehensive_metrics["training_probe_auc_max"]

        summary.append("\n🎯 TRAINING STATISTICS:")
        summary.append(f"  Probe Training AUC:      {mean_auc:.3f} ± {std_auc:.3f}")
        summary.append(f"  Best Training AUC:       {max_auc:.3f}")

    # Optimization Statistics
    if "optimization_configs_tested" in comprehensive_metrics:
        num_configs = comprehensive_metrics["optimization_configs_tested"]
        best_score = comprehensive_metrics.get("validation_score", DEFAULT_SCORE)

        summary.append("\n⚙️ OPTIMIZATION STATISTICS:")
        summary.append(f"  Configurations Tested:   {num_configs}")
        summary.append(f"  Best Validation Score:   {best_score:.3f}")

    # Overall Assessment
    if "overall_effectiveness" in comprehensive_metrics:
        effectiveness = comprehensive_metrics["overall_effectiveness"]

        summary.append("\n🏆 OVERALL ASSESSMENT:")
        if effectiveness > EFFECTIVENESS_HIGH:
            assessment = "Highly Effective"
        elif effectiveness > EFFECTIVENESS_MODERATE:
            assessment = "Moderately Effective"
        elif effectiveness > EFFECTIVENESS_SLIGHT:
            assessment = "Slightly Effective"
        else:
            assessment = "Minimal Effect"

        summary.append(f"  Steering Effectiveness:  {assessment} ({effectiveness:+.3f})")

    summary.append("=" * SEPARATOR_WIDTH_STANDARD)

    return "\n".join(summary)
