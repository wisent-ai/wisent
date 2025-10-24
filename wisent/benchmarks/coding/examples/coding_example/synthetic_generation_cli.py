"""
Synthetic data generation using wisent CLI command.

This file demonstrates the CLI equivalent of the synthetic_generation.py script.
Instead of using Python code directly, it shows how to use the wisent CLI to generate
synthetic contrastive pairs for augmenting training data.

CLI Command Equivalent:
----------------------
wisent generate-synthetic \
  model meta-llama/Llama-3.2-1B-Instruct \
  loader task_interface \
  task gsm8k \
  seed_pairs 10 \
  synthetic_count 100 \
  method caa \
  layers 8 \
  aggregation continuation_token \
  device cuda \
  normalize_layers true \
  save_dir ./steering_output/synthetic

Parameters Explained:
--------------------
- model: The HuggingFace model to use
- loader: Data loader type (task_interface for benchmarks)
- task: Benchmark name (gsm8k, truthfulqa_mc1, hle, etc.)
- seed_pairs: Number of seed pairs to use for generation
- synthetic_count: Number of synthetic pairs to generate
- method: Steering method (caa = Contrastive Activation Addition)
- layers: Layer(s) to train on (e.g., "8", "4-6", "5,7,9")
- aggregation: How to aggregate activations (continuation_token, last_token, etc.)
- device: Device to use (cpu, cuda, cuda:0)
- normalize_layers: Whether to normalize activations per layer
- save_dir: Directory to save synthetic pairs

Note: This is a placeholder. The actual wisent CLI command for synthetic data
generation is not yet implemented. This file shows what the interface would
look like when implemented.

Interactive Mode:
----------------
Add 'interactive true' to get a guided wizard:

wisent generate-synthetic model meta-llama/Llama-3.2-1B-Instruct interactive true

Plan-Only Mode:
--------------
Add 'plan-only true' to preview without executing:

wisent generate-synthetic model meta-llama/Llama-3.2-1B-Instruct plan-only true
"""
import os
import subprocess


def get_config():
    """Read configuration from environment variables with defaults."""
    return {
        'benchmark': os.getenv('WISENT_BENCHMARK', 'gsm8k'),
        'model': os.getenv('WISENT_MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        'layers_spec': os.getenv('WISENT_LAYERS_SPEC', '8'),
        'device': os.getenv('WISENT_DEVICE', 'cpu'),
        'save_dir': os.getenv('WISENT_SAVE_DIR', './steering_output'),
    }


def build_cli_command(config):
    """Build the wisent CLI command from config."""
    cmd = [
        'wisent', 'generate-synthetic',
        'model', config['model'],
        'loader', 'task_interface',
        'task', config['benchmark'],
        'seed_pairs', '10',
        'synthetic_count', '100',
        'method', 'caa',
        'layers', config['layers_spec'],
        'aggregation', 'continuation_token',
        'device', config['device'],
        'normalize_layers', 'true',
        'save_dir', f"{config['save_dir']}/synthetic",
    ]
    return cmd


def main():
    """Execute synthetic data generation using wisent CLI."""
    config = get_config()

    print('=' * 80)
    print('STEP 4: Synthetic Data Generation via Wisent CLI')
    print('=' * 80)
    print(f'Benchmark: {config["benchmark"]}')
    print(f'Model: {config["model"]}')
    print(f'Layers: {config["layers_spec"]}')
    print(f'Device: {config["device"]}')
    print('=' * 80)

    print('\n[Placeholder] Synthetic generation CLI not yet implemented')
    print('\nWhen implemented, this would execute:')

    # Build CLI command
    cmd = build_cli_command(config)

    print(' '.join(cmd))
    print('\n' + '=' * 80)

    print('\nThis module will:')
    print('  1. Generate synthetic contrastive pairs from existing data')
    print('  2. Augment training dataset with synthetic examples')
    print('  3. Evaluate classifier on synthetic + real data')
    print('  4. Compare performance with real data only')

    print('\n' + '=' * 80)
    print('SYNTHETIC GENERATION PLACEHOLDER (CLI)')
    print('=' * 80)


if __name__ == '__main__':
    main()
