"""
CLI entry point for the steering optimization pipeline.
"""

import argparse
import json
import logging

from .config import OptimizationConfig


def main():
    """Main entry point for optimization pipeline."""
    parser = argparse.ArgumentParser(description="Run optimization pipeline")
    parser.add_argument("--output-dir", type=str, default="outputs/optimization_pipeline",
                        help="Directory to save outputs (default: outputs/optimization_pipeline)")
    parser.add_argument("--model", type=str, default="realtreetune/rho-1b-sft-GSM8K",
                        help="Model name or path")
    
    # Task specification
    parser.add_argument("--task", type=str, required=True,
                        help=(
                            "Task to optimize for. Can be: "
                            "'refusal' (compliance optimization), "
                            "'personalization' (requires --trait), "
                            "'custom' (requires --custom-evaluator), "
                            "benchmark name (e.g., 'arc_easy', 'gsm8k'), "
                            "or comma-separated benchmarks (e.g., 'arc_easy,gsm8k,hellaswag')"
                        ))
    
    # Trait description (required for --task personalization)
    parser.add_argument("--trait", type=str, default=None,
                        help="Trait description for personalization (required when --task personalization or custom)")
    parser.add_argument("--trait-label", type=str, default="positive",
                        help="Label for trait direction (default: positive)")
    
    # Custom evaluator (required for --task custom)
    parser.add_argument("--custom-evaluator", type=str, default=None,
                        help="Custom evaluator module path (required when --task custom)")
    parser.add_argument("--custom-evaluator-kwargs", type=str, default=None,
                        help="JSON string of kwargs for custom evaluator")
    
    # Evaluation options
    parser.add_argument("--eval-prompts", type=str, default=None,
                        help="Path to custom evaluation prompts JSON")
    parser.add_argument("--num-eval-prompts", type=int, default=30,
                        help="Number of evaluation prompts (default: 30)")
    
    # Optimization parameters
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of optimization trials (default: 200)")
    parser.add_argument("--train-limit", type=int, default=100,
                        help="Training samples limit")
    parser.add_argument("--val-limit", type=int, default=50,
                        help="Validation samples limit")
    parser.add_argument("--test-limit", type=int, default=50,
                        help="Test samples limit")
    parser.add_argument("--num-pairs", type=int, default=50,
                        help="Number of contrastive pairs (default: 50)")
    parser.add_argument("--layer-range", type=str, default="0-24",
                        help="Layer search range (e.g., '0-24' for all layers)")
    args = parser.parse_args()
    
    # Validate --task personalization requires --trait
    if args.task.lower() == "personalization" and not args.trait:
        parser.error("--trait is required when --task personalization")
    
    # Validate --task custom requires --custom-evaluator and --trait
    if args.task.lower() == "custom":
        if not args.custom_evaluator:
            parser.error("--custom-evaluator is required when --task custom")
        if not args.trait:
            parser.error("--trait is required when --task custom (for steering vector generation)")
    
    # Parse layer range
    layer_start, layer_end = map(int, args.layer_range.split("-"))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Parse custom evaluator kwargs if provided
    custom_evaluator_kwargs = None
    if args.custom_evaluator_kwargs:
        custom_evaluator_kwargs = json.loads(args.custom_evaluator_kwargs)

    # Create configuration
    config = OptimizationConfig(
        model_name=args.model,
        train_dataset=args.task,
        val_dataset=args.task,
        test_dataset=args.task,
        trait=args.trait,
        trait_label=args.trait_label,
        eval_prompts=args.eval_prompts,
        num_eval_prompts=args.num_eval_prompts,
        custom_evaluator=args.custom_evaluator,
        custom_evaluator_kwargs=custom_evaluator_kwargs,
        train_limit=args.train_limit,
        contrastive_pairs_limit=args.num_pairs,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        n_trials=args.n_trials,
        layer_search_range=(layer_start, layer_end),
        output_dir=args.output_dir,
    )

    # Import here to avoid circular imports
    from .pipeline import OptimizationPipeline

    # Run optimization
    pipeline = OptimizationPipeline(config)
    try:
        results = pipeline.run_optimization()

        print("Optimization completed!")
        print(f"Best validation score: {results['best_validation_score']:.4f}")
        print(f"Test accuracy: {results['steered_benchmark_metrics']['accuracy']:.4f}")
        print(f"Accuracy improvement: {results['accuracy_improvement']:+.4f}")

    finally:
        pipeline.cleanup_memory()


if __name__ == "__main__":
    main()
