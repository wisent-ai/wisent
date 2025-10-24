"""
Complete pipeline: Train classifier, apply steering, optimize, and generate synthetic data.

This script orchestrates the execution of 4 independent step files:
1. classifier.py - Train classifier on contrastive pairs
2. train_and_steer.py - Apply steering with trained classifier
3. sample_size_optimization.py - Optimize sample sizes (placeholder)
4. synthetic_generation.py - Generate synthetic pairs (placeholder)

Usage:
    # Run all steps with default config
    python ex_coding_paper_pipeline_limited_examples.py

    # Run specific steps with custom config
    python ex_coding_paper_pipeline_limited_examples.py --benchmark gsm8k --model distilgpt2 --steps 1,2

    # Run only classifier training
    python ex_coding_paper_pipeline_limited_examples.py --steps 1
"""
import argparse
import sys
import os
from typing import List, Optional


class PipelineConfig:
    """Configuration for the coding pipeline."""

    def __init__(
        self,
        benchmark: str = "gsm8k",
        model: str = "distilgpt2",
        training_limit: int = 10,
        testing_limit: int = 2,
        layers_spec: str = "4-6",
        steering_strength: float = 1.0,
        save_dir: str = "./steering_output",
        device: str = "cpu",
        steps: Optional[List[int]] = None
    ):
        self.benchmark = benchmark
        self.model = model
        self.training_limit = training_limit
        self.testing_limit = testing_limit
        self.layers_spec = layers_spec
        self.steering_strength = steering_strength
        self.save_dir = save_dir
        self.device = device
        self.steps = steps if steps is not None else [1, 2, 3, 4]

    def __repr__(self):
        return (
            f"PipelineConfig(\n"
            f"  benchmark={self.benchmark},\n"
            f"  model={self.model},\n"
            f"  training_limit={self.training_limit},\n"
            f"  testing_limit={self.testing_limit},\n"
            f"  layers_spec={self.layers_spec},\n"
            f"  steering_strength={self.steering_strength},\n"
            f"  save_dir={self.save_dir},\n"
            f"  device={self.device},\n"
            f"  steps={self.steps}\n"
            f")"
        )


def train_classifier(config: PipelineConfig):
    """Step 1: Train classifier on contrastive pairs."""
    print('=' * 80)
    print('STEP 1: Train Classifier on Contrastive Pairs')
    print('=' * 80)

    # Set environment variables for the step
    os.environ['WISENT_BENCHMARK'] = config.benchmark
    os.environ['WISENT_MODEL'] = config.model
    os.environ['WISENT_TRAINING_LIMIT'] = str(config.training_limit)
    os.environ['WISENT_TESTING_LIMIT'] = str(config.testing_limit)
    os.environ['WISENT_LAYERS_SPEC'] = config.layers_spec
    os.environ['WISENT_DEVICE'] = config.device
    os.environ['WISENT_SAVE_DIR'] = config.save_dir

    # Import and run the classifier module
    from wisent.benchmarks.coding.examples.coding_example import classifier
    classifier.main()

    print('\n' + '=' * 80)
    print('CLASSIFIER TRAINING COMPLETE')
    print('=' * 80)


def apply_steering(config: PipelineConfig):
    """Step 2: Apply steering with trained classifier."""
    print('=' * 80)
    print('STEP 2: Apply Steering')
    print('=' * 80)

    # Set environment variables for the step
    os.environ['WISENT_BENCHMARK'] = config.benchmark
    os.environ['WISENT_MODEL'] = config.model
    os.environ['WISENT_TRAINING_LIMIT'] = str(config.training_limit)
    os.environ['WISENT_TESTING_LIMIT'] = str(config.testing_limit)
    os.environ['WISENT_STEERING_STRENGTH'] = str(config.steering_strength)
    os.environ['WISENT_DEVICE'] = config.device
    os.environ['WISENT_SAVE_DIR'] = config.save_dir

    # Import and run the train_and_steer module
    from wisent.benchmarks.coding.examples.coding_example import train_and_steer
    train_and_steer.main()

    print('\n' + '=' * 80)
    print('STEERING APPLICATION COMPLETE')
    print('=' * 80)


def sample_size_optimization(config: PipelineConfig):
    """Step 3: Optimize sample sizes."""
    print('=' * 80)
    print('STEP 3: Sample Size Optimization')
    print('=' * 80)

    # Set environment variables for the step
    os.environ['WISENT_BENCHMARK'] = config.benchmark
    os.environ['WISENT_MODEL'] = config.model
    os.environ['WISENT_DEVICE'] = config.device
    os.environ['WISENT_SAVE_DIR'] = config.save_dir

    # Import and run the sample_size_optimization module
    from wisent.benchmarks.coding.examples.coding_example import sample_size_optimization as sample_opt
    sample_opt.main()

    print('\n' + '=' * 80)
    print('SAMPLE SIZE OPTIMIZATION COMPLETE')
    print('=' * 80)


def synthetic_generation(config: PipelineConfig):
    """Step 4: Generate synthetic pairs."""
    print('=' * 80)
    print('STEP 4: Synthetic Data Generation')
    print('=' * 80)

    # Set environment variables for the step
    os.environ['WISENT_BENCHMARK'] = config.benchmark
    os.environ['WISENT_MODEL'] = config.model
    os.environ['WISENT_DEVICE'] = config.device
    os.environ['WISENT_SAVE_DIR'] = config.save_dir

    # Import and run the synthetic_generation module
    from wisent.benchmarks.coding.examples.coding_example import synthetic_generation as synth_gen
    synth_gen.main()

    print('\n' + '=' * 80)
    print('SYNTHETIC GENERATION COMPLETE')
    print('=' * 80)


STEP_FUNCTIONS = {
    1: train_classifier,
    2: apply_steering,
    3: sample_size_optimization,
    4: synthetic_generation,
}


def run_pipeline(config: PipelineConfig):
    """Run the complete pipeline with the given configuration."""
    print('\n' + '=' * 80)
    print('CODING PIPELINE - CONFIGURATION')
    print('=' * 80)
    print(config)
    print('=' * 80 + '\n')

    for step_num in config.steps:
        if step_num not in STEP_FUNCTIONS:
            print(f'⚠ Warning: Unknown step {step_num}, skipping...')
            continue

        print('\n')
        STEP_FUNCTIONS[step_num](config)

    print('\n\n')
    print('=' * 80)
    print('PIPELINE COMPLETE')
    print('=' * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the coding paper pipeline with configurable steps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps with default config
  python ex_coding_paper_pipeline_limited_examples.py

  # Run only training and steering
  python ex_coding_paper_pipeline_limited_examples.py --steps 1,2

  # Use different benchmark and model
  python ex_coding_paper_pipeline_limited_examples.py --benchmark hle --model gpt2

  # Adjust training parameters
  python ex_coding_paper_pipeline_limited_examples.py --training-limit 50 --testing-limit 10
        """
    )

    parser.add_argument('--benchmark', type=str, default='gsm8k',
                        help='Benchmark to use (default: gsm8k)')
    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='Model to use (default: distilgpt2)')
    parser.add_argument('--training-limit', type=int, default=10,
                        help='Number of training pairs (default: 10)')
    parser.add_argument('--testing-limit', type=int, default=2,
                        help='Number of testing pairs (default: 2)')
    parser.add_argument('--layers-spec', type=str, default='4-6',
                        help='Layers to train on, e.g., "4-6" or "5,7,9" (default: 4-6)')
    parser.add_argument('--steering-strength', type=float, default=1.0,
                        help='Steering strength multiplier (default: 1.0)')
    parser.add_argument('--save-dir', type=str, default='./steering_output',
                        help='Directory to save outputs (default: ./steering_output)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: cpu, cuda, cuda:0, etc. (default: cpu)')
    parser.add_argument('--steps', type=str, default='1,2,3,4',
                        help='Comma-separated step numbers to run (default: 1,2,3,4)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Parse steps
    try:
        steps = [int(s.strip()) for s in args.steps.split(',')]
    except ValueError:
        print(f'Error: Invalid steps format: {args.steps}')
        print('Steps must be comma-separated numbers, e.g., "1,2" or "1,2,3,4"')
        sys.exit(1)

    # Create config
    config = PipelineConfig(
        benchmark=args.benchmark,
        model=args.model,
        training_limit=args.training_limit,
        testing_limit=args.testing_limit,
        layers_spec=args.layers_spec,
        steering_strength=args.steering_strength,
        save_dir=args.save_dir,
        device=args.device,
        steps=steps
    )

    # Run pipeline
    run_pipeline(config)
