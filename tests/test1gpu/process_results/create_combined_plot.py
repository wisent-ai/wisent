#!/usr/bin/env python3
"""
Create combined plots from all 4 benchmarks (BoolQ, CB, GSM8K, SST2):
- 2x2: Aggregations combined plots
- 2x2: Train size improvement plots
- 2x2: 2D PCA plots
- 5x4: Individual aggregations (rows: aggregation types, columns: benchmarks)
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# Define the base directory for results
base_dir = os.path.join(os.path.dirname(__file__), '../results')

# Define the paths to the aggregation plots
aggregation_plots = {
    'BoolQ': os.path.join(base_dir, 'boolq/boolq_aggregations_combined_plot.png'),
    'CB': os.path.join(base_dir, 'cb/cb_aggregations_combined_plot.png'),
    'GSM8K': os.path.join(base_dir, 'gsm8k/gsm8k_aggregations_combined_plot.png'),
    'SST2': os.path.join(base_dir, 'sst2/sst2_aggregations_combined_plot.png'),
}

# Define the paths to the train_size_improve plots
train_size_plots = {
    'BoolQ': os.path.join(base_dir, 'boolq/boolq_train_size_improve_plot.png'),
    'CB': os.path.join(base_dir, 'cb/cb_train_size_improve_plot.png'),
    'GSM8K': os.path.join(base_dir, 'gsm8k/gsm8k_train_size_improve_plot.png'),
    'SST2': os.path.join(base_dir, 'sst2/sst2_train_size_improve_plot.png'),
}

# Define the paths to the 2D PCA plots
pca_plots = {
    'BoolQ': os.path.join(base_dir, 'boolq/boolq_pca.png'),
    'CB': os.path.join(base_dir, 'cb/cb_pca.png'),
    'GSM8K': os.path.join(base_dir, 'gsm8k/gsm8k_pca.png'),
    'SST2': os.path.join(base_dir, 'sst2/sst2_pca.png'),
}

# Create aggregations combined plot
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Benchmark Aggregations Comparison', fontsize=20, fontweight='bold')
axes_flat = axes.flatten()

for idx, (name, path) in enumerate(aggregation_plots.items()):
    img = imread(path)
    axes_flat[idx].imshow(img)
    axes_flat[idx].set_title(name, fontsize=16, fontweight='bold')
    axes_flat[idx].axis('off')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'subplots/all_benchmarks_aggregations_2x2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to: {output_path}")
plt.close()

# Create train_size_improve combined plot
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Train Size Improvement Comparison', fontsize=20, fontweight='bold')
axes_flat = axes.flatten()

for idx, (name, path) in enumerate(train_size_plots.items()):
    img = imread(path)
    axes_flat[idx].imshow(img)
    axes_flat[idx].set_title(name, fontsize=16, fontweight='bold')
    axes_flat[idx].axis('off')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'subplots/all_benchmarks_train_size_2x2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to: {output_path}")
plt.close()

# Create 2D PCA combined plot
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('2D PCA Comparison', fontsize=20, fontweight='bold')
axes_flat = axes.flatten()

for idx, (name, path) in enumerate(pca_plots.items()):
    img = imread(path)
    axes_flat[idx].imshow(img)
    axes_flat[idx].set_title(name, fontsize=16, fontweight='bold')
    axes_flat[idx].axis('off')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'subplots/all_benchmarks_pca_2x2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to: {output_path}")
plt.close()

# Create 5x4 grid: aggregations (rows) x benchmarks (columns)
aggregation_types = ['choice_token', 'first_token', 'last_token', 'max_pooling', 'mean_pooling']
benchmarks = ['boolq', 'cb', 'gsm8k', 'sst2']
benchmark_names = ['BoolQ', 'CB', 'GSM8K', 'SST2']

fig, axes = plt.subplots(5, 4, figsize=(24, 30))
fig.suptitle('Individual Aggregations Comparison\n(Rows: Aggregation Types, Columns: Benchmarks)',
             fontsize=24, fontweight='bold')

for row_idx, agg_type in enumerate(aggregation_types):
    for col_idx, (bench, bench_name) in enumerate(zip(benchmarks, benchmark_names)):
        img_path = os.path.join(base_dir, f'{bench}/{bench}_aggregation_{agg_type}_plot.png')
        img = imread(img_path)
        axes[row_idx, col_idx].imshow(img)

        # Add title only to first row (benchmark names) and first column (aggregation types)
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(bench_name, fontsize=18, fontweight='bold')
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(agg_type.replace('_', ' ').title(),
                                               fontsize=16, fontweight='bold', rotation=90, labelpad=20)

        axes[row_idx, col_idx].axis('off')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'subplots/all_benchmarks_aggregations_5x4.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to: {output_path}")
