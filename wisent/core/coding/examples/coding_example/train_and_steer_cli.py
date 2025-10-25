"""
Apply steering vectors using wisent CLI command.

This file demonstrates the CLI equivalent of the train_and_steer.py script.
Instead of using Python code directly, it shows how to use the wisent CLI to apply
steering vectors to a model and generate steered outputs.

CLI Command Equivalent:
----------------------
wisent steer \
  model meta-llama/Llama-3.2-1B-Instruct \
  steering_vectors ./steering_output/steering_vectors.json \
  loader task_interface \
  task gsm8k \
  testing_limit 3 \
  strength 1.0 \
  device cuda \
  max_new_tokens 50 \
  temperature 0.7

Parameters Explained:
--------------------
- model: The HuggingFace model to use
- steering_vectors: Path to trained steering vectors JSON file
- loader: Data loader type (task_interface for benchmarks)
- task: Benchmark name (gsm8k, truthfulqa_mc1, hle, etc.)
- testing_limit: Number of test examples to generate outputs for
- strength: Steering strength multiplier (1.0 = full strength)
- device: Device to use (cpu, cuda, cuda:0)
- max_new_tokens: Maximum tokens to generate per output
- temperature: Generation temperature (0.0 = deterministic, higher = more random)

Interactive Mode:
----------------
Add 'interactive true' to get a guided wizard:

wisent steer model meta-llama/Llama-3.2-1B-Instruct interactive true

Plan-Only Mode:
--------------
Add 'plan-only true' to preview without executing:

wisent steer model meta-llama/Llama-3.2-1B-Instruct plan-only true
"""
import os
import subprocess


def get_config():
    """Read configuration from environment variables with defaults."""
    return {
        'benchmark': os.getenv('WISENT_BENCHMARK', 'gsm8k'),
        'model': os.getenv('WISENT_MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        'testing_limit': int(os.getenv('WISENT_TESTING_LIMIT', '3')),
        'steering_strength': float(os.getenv('WISENT_STEERING_STRENGTH', '1.0')),
        'device': os.getenv('WISENT_DEVICE', 'cpu'),
        'save_dir': os.getenv('WISENT_SAVE_DIR', './steering_output'),
    }


def build_cli_command(config):
    """Build the wisent CLI command from config."""
    cmd = [
        'wisent', 'steer',
        'model', config['model'],
        'steering_vectors', f"{config['save_dir']}/steering_vectors.json",
        'loader', 'task_interface',
        'task', config['benchmark'],
        'testing_limit', str(config['testing_limit']),
        'strength', str(config['steering_strength']),
        'device', config['device'],
        'max_new_tokens', '50',
        'temperature', '0.7',
    ]
    return cmd


def main():
    """Execute steering application using wisent CLI."""
    config = get_config()

    print('=' * 80)
    print('STEP 2: Apply Steering via Wisent CLI')
    print('=' * 80)
    print(f'Benchmark: {config["benchmark"]}')
    print(f'Model: {config["model"]}')
    print(f'Steering strength: {config["steering_strength"]}')
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
        print('STEERING APPLICATION COMPLETE (via CLI)')
        print('=' * 80)
    else:
        print('\n' + '=' * 80)
        print(f'STEERING APPLICATION FAILED (exit code: {result.returncode})')
        print('=' * 80)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
