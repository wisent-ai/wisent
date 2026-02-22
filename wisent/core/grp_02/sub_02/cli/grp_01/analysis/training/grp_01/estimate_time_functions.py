"""Benchmark data and runtime estimation for unified goodness training."""


#!/usr/bin/env python3
"""
Estimate runtime for train-unified-goodness command.

This script provides accurate time estimates based on:
- Benchmark loading times (from measured data)
- Benchmark dataset sizes (number of samples available)
- Activation collection time per pair
- Evaluation time per pair per scale
"""

from __future__ import annotations

import argparse
import sys


# Measured benchmark loading times (seconds) from lm-eval-harness
BENCHMARK_LOAD_TIMES = {
    # HIGH priority (fast loading)
    'cb': 11.0, 'copa': 11.4, 'multirc': 11.6, 'wic': 11.7, 'wsc': 11.1,
    'truthfulqa_mc1': 12.3, 'truthfulqa_mc2': 10.6, 'truthfulqa_gen': 10.4,
    'hellaswag': 12.2, 'piqa': 11.1, 'winogrande': 11.0, 'openbookqa': 13.5,
    'logiqa': 9.7, 'logiqa2': 9.7, 'wsc273': 9.8, 'coqa': 11.3, 'boolq': 11.6,
    'race': 9.9, 'webqs': 13.0, 'mutual': 9.9, 'mmlu': 9.5, 'arc_easy': 10.4,
    'arc_challenge': 10.8, 'sciq': 12.7, 'gsm8k': 12.1, 'math_qa': 12.5,
    'asdiv': 9.5, 'humaneval': 12.5, 'mbpp': 13.1, 'toxigen': 12.4,
    'pubmedqa': 10.6, 'wikitext': 11.0, 'prost': 11.3, 'mmmlu': 12.4,
    'DEFAULT_HIGH': 11.0,
    
    # MEDIUM priority (moderate loading)
    'record': 20.2, 'swag': 16.2, 'drop': 16.6, 'squad2': 16.4, 'triviaqa': 25.6,
    'naturalqs': 13.6, 'headqa_en': 30.8, 'qasper': 29.4, 'qa4mre_2013': 47.8,
    'ai2_arc': 33.0, 'social_iqa': 19.3, 'medqa_4options': 18.9, 'lambada': 34.4,
    'lambada_cloze': 32.2, 'lambada_multilingual': 59.2, 'unscramble': 59.8,
    'DEFAULT_MEDIUM': 30.0,
    
    # LOW priority (slow loading)
    'glue': 129.8, 'superglue': 169.2, 'crows_pairs': 76.6, 'hendrycks_ethics': 65.4,
    'anli': 75.2, 'xnli_en': 210.6, 'xcopa': 91.6, 'xstorycloze_en': 66.4,
    'xwinograd_en': 65.2, 'paws_en': 103.8, 'mgsm': 76.1, 'belebele': 157.9,
    'hendrycks_math': 69.5, 'blimp': 209.5, 'big_bench': 170.0,
    'DEFAULT_LOW': 100.0,
}

# Estimated number of samples per benchmark (test/validation set sizes)
# These are approximate sizes of the datasets we can generate pairs from
BENCHMARK_SIZES = {
    # Reasoning benchmarks
    'hellaswag': 10042, 'piqa': 1838, 'winogrande': 1267, 'openbookqa': 500,
    'copa': 100, 'wsc': 104, 'wsc273': 273, 'swag': 20006,
    'logiqa': 651, 'logiqa2': 1572, 'boolq': 3270,
    
    # Knowledge/QA benchmarks
    'arc_easy': 2376, 'arc_challenge': 1172, 'sciq': 1000,
    'mmlu': 14042, 'triviaqa': 17944, 'webqs': 2032,
    'naturalqs': 3610, 'coqa': 500, 'race': 1045,
    
    # Math benchmarks
    'gsm8k': 1319, 'math_qa': 2985, 'asdiv': 2096,
    'math': 5000, 'math500': 500, 'hendrycks_math': 5000,
    
    # Truthfulness
    'truthfulqa_mc1': 817, 'truthfulqa_mc2': 817, 'truthfulqa_gen': 817,
    
    # Coding
    'humaneval': 164, 'mbpp': 500, 'humaneval_plus': 164, 'mbpp_plus': 399,
    
    # NLI/Inference
    'cb': 56, 'rte': 277, 'wnli': 146, 'qnli': 5463,
    'mrpc': 408, 'qqp': 40430, 'sst2': 872,
    'multirc': 953, 'wic': 638, 'record': 10000,
    
    # Other
    'toxigen': 940, 'pubmedqa': 500, 'medqa_4options': 1273,
    'mutual': 886, 'prost': 18736, 'drop': 9536,
    'social_iqa': 1954, 'squad2': 11873,
    
    # Large benchmarks
    'anli': 3200, 'xnli_en': 5010, 'belebele': 900,
    'glue': 67350, 'superglue': 13368, 'big_bench': 50000,
    
    # Default sizes by priority
    'DEFAULT_HIGH': 1000,
    'DEFAULT_MEDIUM': 2000,
    'DEFAULT_LOW': 5000,
}

# Time estimates for different operations (seconds)
TIME_ESTIMATES = {
    # Model loading (one-time cost)
    'model_load_gpu': 90,      # Loading quantized 8B model on GPU
    'model_load_cpu': 300,     # Loading on CPU (much slower)
    
    # Per-pair times on GPU (g6e.xlarge with Qwen3-8B-GPTQ-Int4)
    'pair_generation': 0.5,    # Generating contrastive pair from benchmark
    'activation_collection': 1.2,  # Forward pass to collect activations
    'evaluation_per_scale': 1.5,   # Inference + evaluation per pair per scale
    
    # Vector training (CAA) - very fast, constant time
    'vector_training': 5,
    
    # Overhead per benchmark
    'benchmark_setup': 2,      # Initial setup per benchmark
}




def estimate_runtime(
    benchmarks: dict,  # {name: config} dict of selected benchmarks
    num_eval_scales: int,
    train_ratio: float,
    skip_evaluation: bool,
    device: str,
    cap_pairs_per_benchmark: int | None = None,  # Optional cap (random sampling)
) -> dict:
    """
    Estimate total runtime for unified goodness training.
    
    Now uses actual benchmark sizes (80% train / 20% eval split).
    
    Returns dict with breakdown of time estimates.
    """
    results = {}
    
    # 1. Model loading (one-time)
    if device == 'cpu' or device == 'auto':
        from wisent.core.utils import resolve_default_device
        actual_device = resolve_default_device() if device == 'auto' else device
        model_time = TIME_ESTIMATES['model_load_cpu'] if actual_device == 'cpu' else TIME_ESTIMATES['model_load_gpu']
    else:
        model_time = TIME_ESTIMATES['model_load_gpu']
    results['model_loading'] = model_time
    
    # 2. Calculate per-benchmark data
    total_load_time = 0
    total_pairs = 0
    total_train_pairs = 0
    total_eval_pairs = 0
    benchmark_details = []
    
    for name, config in benchmarks.items():
        priority = config.get('priority', 'medium')
        load_time = get_benchmark_load_time(name, priority)
        size = get_benchmark_size(name, priority)
        
        # Apply optional limit
        if cap_pairs_per_benchmark:
            size = min(size, cap_pairs_per_benchmark)
        
        train_size = int(size * train_ratio)
        eval_size = size - train_size
        
        total_load_time += load_time
        total_pairs += size
        total_train_pairs += train_size
        total_eval_pairs += eval_size
        
        benchmark_details.append({
            'name': name,
            'priority': priority,
            'load_time': load_time,
            'total_pairs': size,
            'train_pairs': train_size,
            'eval_pairs': eval_size,
        })
    
    results['benchmark_details'] = benchmark_details
    results['num_benchmarks'] = len(benchmarks)
    
    # Benchmark loading
    results['benchmark_loading'] = total_load_time
    
    # Pair generation time
    pair_gen_time = total_pairs * TIME_ESTIMATES['pair_generation']
    results['pair_generation'] = pair_gen_time
    
    # Benchmark setup overhead
    benchmark_setup_time = len(benchmarks) * TIME_ESTIMATES['benchmark_setup']
    results['benchmark_setup'] = benchmark_setup_time
    
    # 3. Activation collection (training pairs only)
    activation_time = total_train_pairs * TIME_ESTIMATES['activation_collection']
    results['activation_collection'] = activation_time
    results['total_train_pairs'] = total_train_pairs
    results['total_pairs'] = total_pairs
    
    # 4. Vector training (very fast)
    results['vector_training'] = TIME_ESTIMATES['vector_training']
    
    # 5. Evaluation (if enabled)
    if skip_evaluation:
        results['evaluation'] = 0
        results['total_eval_pairs'] = 0
    else:
        # Eval time = eval_pairs * num_scales * time_per_eval
        total_eval_samples = total_eval_pairs * num_eval_scales
        eval_time = total_eval_samples * TIME_ESTIMATES['evaluation_per_scale']
        results['evaluation'] = eval_time
        results['total_eval_pairs'] = total_eval_pairs
        results['total_eval_samples'] = total_eval_samples
    
    # Total
    results['total_seconds'] = (
        results['model_loading'] +
        results['benchmark_loading'] +
        results['pair_generation'] +
        results['benchmark_setup'] +
        results['activation_collection'] +
        results['vector_training'] +
        results['evaluation']
    )
    
    results['total_minutes'] = results['total_seconds'] / 60
    results['total_hours'] = results['total_seconds'] / 3600
    
    return results


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


