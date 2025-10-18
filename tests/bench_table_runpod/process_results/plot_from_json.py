#!/usr/bin/env python3
"""
Read results from JSON files and create plots without binomial errors.

This script reads pre-computed results from JSON files in the results folders
and creates clean plots using only the mean accuracy (without binomial error bars).
Optionally uses std_accuracy from the 10 runs as error bars instead.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def load_aggregation_results(benchmark: str, base_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load aggregation results for a benchmark from JSON files.

    Returns:
        Dict[aggregation_name] -> Dict[layer_name] -> mean_accuracy
    """
    aggregation_types = ['choice_token', 'first_token', 'last_token', 'max_pooling', 'mean_pooling']
    results = {}

    for agg_type in aggregation_types:
        json_path = os.path.join(base_dir, benchmark, f'{benchmark}_aggregation_{agg_type}_results.json')

        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping...")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract mean_accuracy for each layer
        layer_accuracies = {}
        for layer_name, layer_data in data['results_by_layer'].items():
            layer_accuracies[layer_name] = layer_data['mean_accuracy']

        results[agg_type] = layer_accuracies

    return results


def load_train_size_results(benchmark: str, base_dir: str) -> Dict[int, Dict[str, float]]:
    """
    Load train size results for a benchmark from JSON files.
    Auto-detects the aggregation method used for each benchmark.

    Returns:
        Dict[k] -> Dict[layer_name] -> mean_accuracy
    """
    # Each benchmark used a different aggregation method
    aggregation_map = {
        'boolq': 'mean_pooling',
        'cb': 'last_token',
        'gsm8k': 'last_token',
        'sst2': 'choice_token',
    }

    aggregation = aggregation_map.get(benchmark, 'mean_pooling')
    k_values = [5, 10, 20, 50, 100, 250]
    results = {}

    for k in k_values:
        json_path = os.path.join(base_dir, benchmark, f'{benchmark}_train_size_k{k}_{aggregation}_results.json')

        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping...")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract mean_accuracy for each layer
        layer_accuracies = {}
        for layer_name, layer_data in data['results_by_layer'].items():
            layer_accuracies[layer_name] = layer_data['mean_accuracy']

        results[k] = layer_accuracies

    return results


def plot_aggregations_combined(benchmark: str, results: Dict[str, Dict[str, float]], output_path: str):
    """
    Plot all aggregation methods for one benchmark on the same chart (without binomial errors).
    """
    plt.figure(figsize=(14, 8))

    # Plot each aggregation method
    for agg_name in sorted(results.keys()):
        layer_accuracies = results[agg_name]

        # Sort by layer number
        layers = sorted([int(layer) for layer in layer_accuracies.keys()])
        accuracies = [layer_accuracies[str(layer)] for layer in layers]

        # Convert to percentage
        accuracies_pct = [acc * 100 for acc in accuracies]

        # Plot line without error bars
        plt.plot(
            layers,
            accuracies_pct,
            marker='o',
            linewidth=2,
            markersize=6,
            label=agg_name.replace('_', ' ').title(),
            alpha=0.8
        )

    plt.xlabel("Layer Number", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.title(f"Classifier Performance: All Aggregations ({benchmark.upper()})", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=1)

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Aggregations plot saved: {output_path}")


def plot_train_size_combined(benchmark: str, results: Dict[int, Dict[str, float]], output_path: str):
    """
    Plot train size curves for one benchmark (without binomial errors).
    """
    # Aggregation method used for each benchmark
    aggregation_map = {
        'boolq': 'mean_pooling',
        'cb': 'last_token',
        'gsm8k': 'last_token',
        'sst2': 'choice_token',
    }
    aggregation = aggregation_map.get(benchmark, 'mean_pooling')
    agg_display = aggregation.replace('_', ' ').title()

    plt.figure(figsize=(14, 8))

    # Plot each k value
    for k in sorted(results.keys()):
        layer_accuracies = results[k]

        # Sort by layer number
        layers = sorted([int(layer) for layer in layer_accuracies.keys()])
        accuracies = [layer_accuracies[str(layer)] for layer in layers]

        # Convert to percentage
        accuracies_pct = [acc * 100 for acc in accuracies]

        # Plot line without error bars
        plt.plot(
            layers,
            accuracies_pct,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{k*2} training samples',
            alpha=0.8
        )

    plt.xlabel("Layer Number", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.title(f"Classifier Performance: Train Size ({benchmark.upper()}, {agg_display})", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=1)

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Train size plot saved: {output_path}")


def create_2x2_combined_plot(plot_type: str, benchmarks: List[str], base_dir: str, output_dir: str):
    """
    Create 2x2 combined plots from individual clean plots.

    Args:
        plot_type: 'aggregations' or 'train_size'
        benchmarks: List of benchmark names
        base_dir: Base directory for results
        output_dir: Output directory for plots
    """
    from matplotlib.image import imread

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    benchmark_names = {'boolq': 'BoolQ', 'cb': 'CB', 'gsm8k': 'GSM8K', 'sst2': 'SST2'}

    if plot_type == 'aggregations':
        fig.suptitle('Benchmark Aggregations Comparison (Clean)', fontsize=20, fontweight='bold')
        filename_pattern = '{}_aggregations_clean.png'
    else:
        fig.suptitle('Train Size Improvement Comparison (Clean)', fontsize=20, fontweight='bold')
        filename_pattern = '{}_train_size_clean.png'

    axes_flat = axes.flatten()

    for idx, benchmark in enumerate(benchmarks):
        img_path = os.path.join(output_dir, 'plots', filename_pattern.format(benchmark))

        if os.path.exists(img_path):
            img = imread(img_path)
            axes_flat[idx].imshow(img)
            axes_flat[idx].set_title(benchmark_names[benchmark], fontsize=16, fontweight='bold')
        else:
            axes_flat[idx].text(0.5, 0.5, f'{benchmark}\nNot Available',
                              ha='center', va='center', fontsize=14)

        axes_flat[idx].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'subplots', f'all_benchmarks_{plot_type}_2x2_clean.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"2x2 combined plot saved: {output_path}")


def main():
    """Generate all plots from JSON data."""

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '../results')
    output_dir = script_dir

    benchmarks = ['boolq', 'cb', 'gsm8k', 'sst2']

    print("=" * 80)
    print("GENERATING PLOTS FROM JSON DATA (NO BINOMIAL ERRORS)")
    print("=" * 80)

    # Generate aggregation plots for each benchmark
    print("\n--- Aggregations Plots ---")
    for benchmark in benchmarks:
        print(f"\nProcessing {benchmark}...")
        results = load_aggregation_results(benchmark, base_dir)

        if not results:
            print(f"  No results found for {benchmark}")
            continue

        output_path = os.path.join(output_dir, 'plots', f'{benchmark}_aggregations_clean.png')
        plot_aggregations_combined(benchmark, results, output_path)

    # Generate train size plots for each benchmark
    print("\n--- Train Size Plots ---")
    for benchmark in benchmarks:
        print(f"\nProcessing {benchmark}...")
        results = load_train_size_results(benchmark, base_dir)

        if not results:
            print(f"  No results found for {benchmark}")
            continue

        output_path = os.path.join(output_dir, 'plots', f'{benchmark}_train_size_clean.png')
        plot_train_size_combined(benchmark, results, output_path)

    # Create 2x2 combined plots
    print("\n--- Creating 2x2 Combined Plots ---")
    create_2x2_combined_plot('aggregations', benchmarks, base_dir, output_dir)
    create_2x2_combined_plot('train_size', benchmarks, base_dir, output_dir)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
