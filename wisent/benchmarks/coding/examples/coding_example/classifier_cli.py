"""
Train classifier using wisent CLI command.

This file demonstrates the CLI equivalent of the classifier.py training script.
Instead of using Python code directly, it shows how to use the wisent CLI to train
steering vectors (classifier) on contrastive pairs.

CLI Command Equivalent:
----------------------
wisent train \
  model meta-llama/Llama-3.2-1B-Instruct \
  loader task_interface \
  task gsm8k \
  training_limit 10 \
  testing_limit 2 \
  method caa \
  layers 8 \
  aggregation continuation_token \
  device cuda \
  normalize_layers true \
  save_dir ./steering_output

Parameters Explained:
--------------------
- model: The HuggingFace model to use
- loader: Data loader type (task_interface for benchmarks)
- task: Benchmark name (gsm8k, truthfulqa_mc1, hle, etc.)
- training_limit: Number of training pairs to use
- testing_limit: Number of testing pairs to use
- method: Steering method (caa = Contrastive Activation Addition)
- layers: Layer(s) to train on (e.g., "8", "4-6", "5,7,9")
- aggregation: How to aggregate activations (continuation_token, last_token, etc.)
- device: Device to use (cpu, cuda, cuda:0)
- normalize_layers: Whether to normalize activations per layer
- save_dir: Directory to save trained steering vectors

Interactive Mode:
----------------
Add 'interactive true' to get a guided wizard:

wisent train model meta-llama/Llama-3.2-1B-Instruct interactive true

Plan-Only Mode:
--------------
Add 'plan-only true' to preview without executing:

wisent train model meta-llama/Llama-3.2-1B-Instruct plan-only true
"""
import os
import subprocess


def get_config():
    """Read configuration from environment variables with defaults."""
    return {
        'benchmark': os.getenv('WISENT_BENCHMARK', 'gsm8k'),
        'model': os.getenv('WISENT_MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        'training_limit': int(os.getenv('WISENT_TRAINING_LIMIT', '10')),
        'testing_limit': int(os.getenv('WISENT_TESTING_LIMIT', '2')),
        'layers_spec': os.getenv('WISENT_LAYERS_SPEC', '8'),
        'device': os.getenv('WISENT_DEVICE', 'cpu'),
        'save_dir': os.getenv('WISENT_SAVE_DIR', './steering_output'),
    }


def build_cli_command(config):
    """Build the wisent CLI command from config."""
    cmd = [
        'wisent', 'train',
        'model', config['model'],
        'loader', 'task_interface',
        'task', config['benchmark'],
        'training_limit', str(config['training_limit']),
        'testing_limit', str(config['testing_limit']),
        'method', 'caa',
        'layers', config['layers_spec'],
        'aggregation', 'continuation_token',
        'device', config['device'],
        'normalize_layers', 'true',
        'save_dir', config['save_dir'],
    ]
    return cmd


def main():
    """Execute classifier training using wisent CLI."""
    config = get_config()

    print('=' * 80)
    print('STEP 1: Train Classifier via Wisent CLI')
    print('=' * 80)
    print(f'Benchmark: {config["benchmark"]}')
    print(f'Model: {config["model"]}')
    print(f'Layers: {config["layers_spec"]}')
    print(f'Device: {config["device"]}')
    print('=' * 80)

    # Build CLI command
    cmd = build_cli_command(config)

    print('\nExecuting CLI command:')
    print(' '.join(cmd))
    print('\n' + '=' * 80)

    # Execute the command
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print('\n' + '=' * 80)
        print('CLASSIFIER TRAINING COMPLETE (via CLI)')
        print('=' * 80)
    else:
        print('\n' + '=' * 80)
        print(f'CLASSIFIER TRAINING FAILED (exit code: {result.returncode})')
        print('=' * 80)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
