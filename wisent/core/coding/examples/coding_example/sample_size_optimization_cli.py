"""
Sample size optimization using wisent CLI command.

This file demonstrates the CLI equivalent of the sample_size_optimization.py script.
Instead of using Python code directly, it shows how to use the wisent CLI to optimize
training sample sizes for classifier performance.

CLI Command Equivalent:
----------------------
wisent optimize-samples \
  model meta-llama/Llama-3.2-1B-Instruct \
  loader task_interface \
  task gsm8k \
  method caa \
  layers 8 \
  aggregation continuation_token \
  device cuda \
  normalize_layers true \
  sample_sizes 10,50,100,500 \
  save_dir ./steering_output/optimization

Parameters Explained:
--------------------
- model: The HuggingFace model to use
- loader: Data loader type (task_interface for benchmarks)
- task: Benchmark name (gsm8k, truthfulqa_mc1, hle, etc.)
- method: Steering method (caa = Contrastive Activation Addition)
- layers: Layer(s) to train on (e.g., "8", "4-6", "5,7,9")
- aggregation: How to aggregate activations (continuation_token, last_token, etc.)
- device: Device to use (cpu, cuda, cuda:0)
- normalize_layers: Whether to normalize activations per layer
- sample_sizes: Comma-separated list of training sample sizes to test
- save_dir: Directory to save optimization results

Note: This is a placeholder. The actual wisent CLI command for sample size
optimization is not yet implemented. This file shows what the interface would
look like when implemented.

Interactive Mode:
----------------
Add 'interactive true' to get a guided wizard:

wisent optimize-samples model meta-llama/Llama-3.2-1B-Instruct interactive true

Plan-Only Mode:
--------------
Add 'plan-only true' to preview without executing:

wisent optimize-samples model meta-llama/Llama-3.2-1B-Instruct plan-only true
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
        'wisent', 'optimize-samples',
        'model', config['model'],
        'loader', 'task_interface',
        'task', config['benchmark'],
        'method', 'caa',
        'layers', config['layers_spec'],
        'aggregation', 'continuation_token',
        'device', config['device'],
        'normalize_layers', 'true',
        'sample_sizes', '10,50,100,500',
        'save_dir', f"{config['save_dir']}/optimization",
    ]
    return cmd


def main():
    """Execute sample size optimization using wisent CLI."""
    config = get_config()

    print('=' * 80)
    print('STEP 3: Sample Size Optimization via Wisent CLI')
    print('=' * 80)
    print(f'Benchmark: {config["benchmark"]}')
    print(f'Model: {config["model"]}')
    print(f'Layers: {config["layers_spec"]}')
    print(f'Device: {config["device"]}')
    print('=' * 80)

    print('\n[Placeholder] Sample size optimization CLI not yet implemented')
    print('\nWhen implemented, this would execute:')

    # Build CLI command
    cmd = build_cli_command(config)

    print(' '.join(cmd))
    print('\n' + '=' * 80)

    print('\nThis module will:')
    print('  1. Test different training set sizes (10, 50, 100, 500)')
    print('  2. Evaluate classifier performance vs sample size')
    print('  3. Determine optimal sample size for target accuracy')
    print('  4. Generate sample efficiency curves')

    print('\n' + '=' * 80)
    print('SAMPLE SIZE OPTIMIZATION PLACEHOLDER (CLI)')
    print('=' * 80)


if __name__ == '__main__':
    main()
