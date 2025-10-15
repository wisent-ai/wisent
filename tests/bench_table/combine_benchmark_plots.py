#!/usr/bin/env python3
"""
Combine plots from all benchmarks into a single visualization.

Creates an 8x4 grid where:
- Rows: Different plot types (8 plots per benchmark)
- Columns: Different benchmarks (GSM8K, BoolQ, SST2, CB)

Each benchmark should have 8 plots (in order):
1. Combined aggregations plot
2. Train size improvement plot
3. Aggregation: continuation_token
4. Aggregation: last_token
5. Aggregation: first_token
6. Aggregation: mean_pooling
7. Aggregation: choice_token
8. Aggregation: max_pooling
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_image_safe(image_path: Path) -> np.ndarray | None:
    """
    Load an image, return None if file doesn't exist.

    Args:
        image_path: Path to the image file

    Returns:
        Image array or None if file doesn't exist
    """
    if not image_path.exists():
        print(f"⚠️  Warning: Image not found: {image_path}")
        return None

    try:
        img = mpimg.imread(image_path)
        return img
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return None


def create_placeholder_image(width: int = 800, height: int = 600) -> np.ndarray:
    """
    Create a placeholder image for missing plots.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Placeholder image array (gray background with "Not Available" text)
    """
    # Create gray image
    img = np.ones((height, width, 3)) * 0.9
    return img


def combine_all_benchmark_plots(
    benchmark_configs: Dict[str, Dict[str, str]],
    output_path: str = "tests/bench_table/combined_plots/all_benchmarks_combined.png",
    figsize: tuple[int, int] = (24, 32),
    dpi: int = 150,
):
    """
    Combine all benchmark plots into a single 8x4 grid.

    Args:
        benchmark_configs: Dictionary mapping benchmark names to their plot configurations
            Format: {
                "GSM8K": {
                    "continuation_token": "path/to/plot.png",
                    "last_token": "path/to/plot.png",
                    ...
                },
                "BoolQ": {...},
                ...
            }
        output_path: Path to save the combined plot
        figsize: Figure size in inches (width, height)
        dpi: DPI for saving the figure
    """

    print("\n" + "=" * 80)
    print("COMBINING BENCHMARK PLOTS")
    print("=" * 80)

    # Define the order of plot types (rows)
    plot_types = [
        "combined_aggregations",
        "train_size_improvement",
        "continuation_token",
        "last_token",
        "first_token",
        "mean_pooling",
        "choice_token",
        "max_pooling",
    ]

    # Define the order of benchmarks (columns)
    benchmarks = ["GSM8K", "BoolQ", "SST2", "CB"]

    num_rows = len(plot_types)
    num_cols = len(benchmarks)

    print(f"\nCreating {num_rows}x{num_cols} grid...")
    print(f"  Rows (plot types): {num_rows}")
    print(f"  Columns (benchmarks): {num_cols}")
    print(f"  Total subplots: {num_rows * num_cols}")

    # Create figure with subplots
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        gridspec_kw={'hspace': 0.3, 'wspace': 0.2}
    )

    # Load and display images
    loaded_count = 0
    missing_count = 0

    for row_idx, plot_type in enumerate(plot_types):
        for col_idx, benchmark in enumerate(benchmarks):
            ax = axes[row_idx, col_idx]

            # Get the image path from config
            # Convert benchmark name to match config keys (all uppercase)
            benchmark_key = benchmark.upper()
            if benchmark_key in benchmark_configs and plot_type in benchmark_configs[benchmark_key]:
                image_path = Path(benchmark_configs[benchmark_key][plot_type])
                img = load_image_safe(image_path)
            else:
                print(f"⚠️  Warning: No config for {benchmark}/{plot_type}")
                img = None

            # Display image or placeholder
            if img is not None:
                ax.imshow(img)
                loaded_count += 1
            else:
                # Show placeholder for missing images
                placeholder = create_placeholder_image()
                ax.imshow(placeholder)
                ax.text(
                    0.5, 0.5,
                    f"Not Available\n{benchmark}\n{plot_type}",
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=10,
                    color='gray'
                )
                missing_count += 1

            # Remove axis ticks
            ax.axis('off')

            # Add title only for top row (benchmark names)
            if row_idx == 0:
                ax.set_title(benchmark, fontsize=14, fontweight='bold', pad=10)

            # Add row labels on the left
            if col_idx == 0:
                # Format plot type name for display
                label = plot_type.replace('_', ' ').title()
                ax.text(
                    -0.1, 0.5,
                    label,
                    ha='right', va='center',
                    transform=ax.transAxes,
                    fontsize=11,
                    rotation=0,
                    fontweight='bold'
                )

    # Add overall title
    fig.suptitle(
        'Benchmark Performance Analysis - All Aggregation Methods',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving combined plot...")
    print(f"  Output: {output_path}")
    print(f"  Loaded images: {loaded_count}")
    print(f"  Missing images: {missing_count}")

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Successfully saved combined plot to {output_path}")
    print("=" * 80)

    return output_path


def get_default_benchmark_configs() -> Dict[str, Dict[str, str]]:
    """
    Get default paths for all benchmark plots.

    Returns:
        Dictionary with default plot paths for each benchmark
    """
    base_dir = Path("tests/bench_table")

    configs = {}

    for benchmark in ["gsm8k", "boolq", "sst2", "cb"]:
        plots_dir = base_dir / benchmark / "plots"

        configs[benchmark.upper()] = {
            "continuation_token": str(plots_dir / "aggregation_continuation_token_plot.png"),
            "last_token": str(plots_dir / "aggregation_last_token_plot.png"),
            "first_token": str(plots_dir / "aggregation_first_token_plot.png"),
            "mean_pooling": str(plots_dir / "aggregation_mean_pooling_plot.png"),
            "choice_token": str(plots_dir / "aggregation_choice_token_plot.png"),
            "max_pooling": str(plots_dir / "aggregation_max_pooling_plot.png"),
            "combined_aggregations": str(plots_dir / "aggregations_combined_plot.png"),
            "train_size_improvement": str(plots_dir / "train_size_improve_plot.png"),
        }

    return configs


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# BENCHMARK PLOT COMBINER")
    print("#" * 80)

    # Get default configurations
    configs = get_default_benchmark_configs()

    print("\nBenchmark configurations:")
    for benchmark, plot_configs in configs.items():
        print(f"\n{benchmark}:")
        for plot_type, path in plot_configs.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  [{exists}] {plot_type}: {path}")

    # Create combined plot
    output_file = combine_all_benchmark_plots(
        benchmark_configs=configs,
        output_path="tests/bench_table/all_benchmarks_combined.png",
        figsize=(24, 32),
        dpi=150,
    )

    print(f"\n{'#' * 80}")
    print("# DONE!")
    print(f"# Output: {output_file}")
    print(f"{'#' * 80}\n")
