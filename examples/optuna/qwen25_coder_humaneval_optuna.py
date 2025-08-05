"""
Qwen2.5-Coder-7B HumanEval Optimization with Optuna

This script demonstrates a coding-focused optimization pipeline for Qwen2.5-Coder-7B
model using HumanEval dataset with Optuna hyperparameter optimization,
SQLite persistence, and WandB experiment tracking.

RECOMMENDED BATCH SIZES:
- Qwen2.5-Coder-7B: batch_size=2-4 (depending on GPU memory)
- Training samples: 50-100 (HumanEval has 164 problems total)
- Validation samples: 30-50 (enough for reliable optimization signal)
- Test samples: 50-100 (comprehensive final evaluation)

STEERING METHODS INVESTIGATED:
1. CAA (Contrastive Activation Addition) - Classic vector steering
2. DAC (Dynamic Activation Control) - Adaptive steering with entropy thresholds

DATASETS (CODING FOCUS):
- Training: humaneval (Python programming problems from OpenAI)
- Validation: humaneval (same dataset for consistency)  
- Test: humaneval (same dataset for testing)

CONTRASTIVE PAIRS GENERATION:
Uses specialized HumanEval extractors that create "obscured correct answer" pairs:
- Correct: Original working code solution
- Incorrect: Syntactically corrupted code (missing tokens, wrong syntax) 
  that "obscures" the correct answer

USAGE:
    # Basic usage with default settings
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_humaneval_optuna.py

    # Custom model path and batch size
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_humaneval_optuna.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --batch-size 2

    # Enable WandB logging
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_humaneval_optuna.py --use-wandb --wandb-project qwen-humaneval-optimization

    # Quick test run
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_humaneval_optuna.py --n-trials 10 --train-limit 30 --val-limit 20

EXPECTED RESULTS:
- Baseline Qwen2.5-Coder-7B typically achieves ~40-60% on HumanEval (challenging dataset)
- Optimal steering can further improve performance on coding tasks
- Over-steering (high α values) typically degrades performance
- Layer selection is crucial - middle layers (16-24) often work best for Qwen

OUTPUTS:
- SQLite database: optuna_studies.db (persistent across runs)
- Results: outputs/qwen25_coder_humaneval_optimization/
- WandB logs: https://wandb.ai/your-project/qwen-humaneval-optimization

MEMORY OPTIMIZATION:
- Uses efficient attention mechanisms
- Caches activations efficiently
- Consider CUDA_VISIBLE_DEVICES to limit GPU usage
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from wisent_guard.core.optuna.optuna_pipeline import OptimizationConfig, OptimizationPipeline


def get_recommended_config_for_qwen25_coder() -> Dict[str, Any]:
    """Get recommended configuration values for Qwen2.5-Coder-7B HumanEval optimization."""
    return {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",  # Qwen2.5-Coder specialized for coding
        "batch_size": 2,  # Conservative for 7B model
        "max_new_tokens": 512,  # Longer for coding tasks - Qwen can handle complex code
        "layer_search_range": (16, 24),  # Qwen has 32 layers (0-31), middle-to-late layers work well for code
        "train_limit": 80,  # Good balance for HumanEval
        "contrastive_pairs_limit": 40,  # Bounded by train_limit
        "val_limit": 40,
        "test_limit": 80,
        "n_trials": 50,  # More trials for better optimization
        "n_startup_trials": 10,  # More random exploration
    }


def create_qwen25_coder_config(args) -> OptimizationConfig:
    """Create optimized configuration for Qwen2.5-Coder-7B HumanEval optimization."""

    # Get base recommendations
    defaults = get_recommended_config_for_qwen25_coder()

    return OptimizationConfig(
        # Model configuration - Qwen2.5-Coder-7B specialized for coding
        model_name=args.model_path or defaults["model_name"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Dataset configuration - Coding focus with HumanEval
        train_dataset="humaneval",  # Python programming problems from OpenAI
        val_dataset="humaneval",    # Same dataset for consistency
        test_dataset="humaneval",   # Same dataset for testing
        
        # Training configuration
        train_limit=args.train_limit or defaults["train_limit"],
        contrastive_pairs_limit=args.contrastive_pairs_limit or defaults["contrastive_pairs_limit"],
        
        # Evaluation configuration
        val_limit=args.val_limit or defaults["val_limit"],
        test_limit=args.test_limit or defaults["test_limit"],
        
        # Layer search configuration - Qwen has 32 layers (0-31)
        # Middle-to-late layers (16-24) typically capture code semantics well
        layer_search_range=args.layer_range or defaults["layer_search_range"],
        
        # Probe type - Fixed to logistic regression
        probe_type="logistic_regression",
        
        # Steering methods - Currently implemented methods
        steering_methods=["caa", "dac"],
        
        # Optuna study configuration
        study_name=args.study_name or "qwen25_coder_humaneval_optimizationv2",
        db_url=f"sqlite:///{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/optuna_studies.db",
        n_trials=args.n_trials or defaults["n_trials"],
        n_startup_trials=args.n_startup_trials or defaults["n_startup_trials"],
        sampler="TPE",  # Tree-structured Parzen Estimator
        pruner="MedianPruner",  # Aggressive pruning for efficiency
        
        # WandB configuration
        wandb_project=args.wandb_project or "qwen25-coder-humaneval-optimization",
        use_wandb=args.use_wandb,
        
        # Generation configuration - Optimized for coding tasks
        batch_size=args.batch_size or defaults["batch_size"],
        max_length=1024,  # Longer for complex coding problems
        max_new_tokens=defaults["max_new_tokens"],
        temperature=0.1,  # Lower temperature for more deterministic code generation
        do_sample=True,
        
        # Performance optimization
        seed=42,
        
        # Output configuration
        output_dir="outputs/qwen25_coder_humaneval_optimization",
        cache_dir="cache/qwen25_coder_humaneval_optimization",
        
        # Search space constraints
        max_layers_to_search=9,  # Search more layers for better coverage
        early_stopping_patience=15,  # More patience for specialized model
    )


class Qwen25CoderHumanEvalPipeline(OptimizationPipeline):
    """Specialized pipeline for Qwen2.5-Coder-7B optimization with HumanEval dataset."""

    def _objective_function(self, trial) -> float:
        """Enhanced objective function for coding tasks with method-specific hyperparameter spaces."""
        try:
            self.logger.info(f"🔬 Trial {trial.number}: Starting HumanEval optimization")

            # Sample layer and probe hyperparameters
            layer_id = trial.suggest_int(
                "layer_id", self.config.layer_search_range[0], self.config.layer_search_range[1]
            )

            # Fixed probe type and regularization
            probe_type = self.config.probe_type  # Always logistic_regression
            probe_c = 1.0  # Default regularization strength

            # Sample steering method and method-specific hyperparameters
            steering_method = trial.suggest_categorical("steering_method", self.config.steering_methods)

            if steering_method == "caa":
                # CAA hyperparameters - adjusted for coding-specialized model
                steering_alpha = trial.suggest_float(  # maps to `strength`
                    "steering_alpha", 0.05, 1.5, step=0.05  # Moderate range for specialized model
                )

                normalization_method = trial.suggest_categorical("normalization_method", ["none", "l2_unit"])

                target_norm = None
                if normalization_method != "none":
                    target_norm = trial.suggest_float("target_norm", 0.7, 1.3, step=0.1)

                steering_params = {
                    "steering_alpha": steering_alpha,
                    "normalization_method": normalization_method,
                    "target_norm": target_norm,
                }

            elif steering_method == "dac":
                # DAC: Dynamic control with entropy-based adaptation
                steering_params = {
                    "base_strength": trial.suggest_float("base_strength", 0.3, 1.2, step=0.05),  # Moderate for coding model
                    "ptop": trial.suggest_float("ptop", 0.25, 0.55, step=0.05),
                    "max_alpha": trial.suggest_float("max_alpha", 0.8, 2.5, step=0.1),  # Reasonable max for coding
                    "entropy_threshold": trial.suggest_float("entropy_threshold", 1.8, 3.5, step=0.1),
                }

            else:
                raise ValueError(f"steering_method: {steering_method} not implemented")

            alpha_str = (
                f"{steering_params.get('steering_alpha', 'N/A'):.3f}"
                if steering_params.get("steering_alpha") is not None
                else "N/A"
            )
            self.logger.info(f"🎯 Trial {trial.number}: {steering_method.upper()} with α={alpha_str} (Layer {layer_id})")

            # Step 1: Train and evaluate probe
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            self.logger.info(f"📊 Trial {trial.number}: Probe {probe_type} AUC = {probe_score:.4f}")

            # Step 2: Train steering method
            steering_instance = self._train_steering_method(trial, steering_method, layer_id, steering_params)

            # Step 3: Evaluate steering on validation set
            validation_accuracy = self._evaluate_steering_on_validation(
                steering_instance, steering_method, layer_id, steering_params
            )

            self.logger.info(f"🎯 Trial {trial.number}: Validation HumanEval accuracy = {validation_accuracy:.4f}")
            trial.report(validation_accuracy, step=1)

            # Enhanced WandB logging with coding-specific metrics
            metrics = {
                "validation_accuracy": validation_accuracy,
                "probe_score": probe_score,
                "method": steering_method,
                "layer": layer_id,
                "task_type": "coding",
                "dataset": "humaneval",
                "model": "qwen2.5-coder-7b",
            }
            self._log_trial_to_wandb(trial, metrics)

            return validation_accuracy

        except Exception as e:
            self.logger.error(f"❌ Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _log_enhanced_results(self, study, final_results):
        """Log enhanced results with HumanEval-specific analysis."""
        self.logger.info("=" * 80)
        self.logger.info("🤖 QWEN2.5-CODER-7B HUMANEVAL OPTIMIZATION RESULTS")
        self.logger.info("=" * 80)

        best_trial = study.best_trial
        best_method = best_trial.params.get("steering_method", "unknown")
        best_alpha = best_trial.params.get("steering_alpha", 0.0)
        best_layer = best_trial.params.get("layer_id", -1)

        self.logger.info(f"🥇 Best Method: {best_method.upper()}")
        self.logger.info(f"📊 Best Layer: {best_layer}")
        self.logger.info(f"⚡ Best Alpha: {best_alpha:.4f}")
        self.logger.info(f"🎯 Best Validation Accuracy: {study.best_value:.4f}")

        baseline_acc = final_results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = final_results["steered_benchmark_metrics"]["accuracy"]
        improvement = final_results["accuracy_improvement"]

        self.logger.info("📈 Test Results on HumanEval:")
        self.logger.info(f"   Baseline:  {baseline_acc:.4f}")
        self.logger.info(f"   Steered:   {steered_acc:.4f}")
        self.logger.info(f"   Improvement: {improvement:+.4f}")

        # Coding-specific insights
        self.logger.info("🔧 HumanEval Task Insights:")
        self.logger.info(f"   - Model: Qwen2.5-Coder-7B (32 layers, specialized for coding)")
        self.logger.info(f"   - Training: HumanEval (OpenAI Python problems)")
        self.logger.info(f"   - Testing: HumanEval (same dataset)")
        self.logger.info(f"   - Industry-standard benchmark for code generation")
        self.logger.info(f"   - Best layer {best_layer} suggests {'early' if best_layer < 11 else 'middle' if best_layer < 22 else 'late'} processing")

        # Performance context
        if baseline_acc > 0.5:
            self.logger.info("🎉 Excellent baseline performance on HumanEval!")
        elif baseline_acc > 0.3:
            self.logger.info("👍 Good baseline performance for HumanEval")
        else:
            self.logger.info("🤔 Lower than expected baseline - check model loading and prompt format")

        # Method-specific insights
        method_trials = [t for t in study.trials if t.params.get("steering_method") == best_method]
        if len(method_trials) > 1:
            method_values = [t.value for t in method_trials if t.value is not None]
            if method_values:
                self.logger.info(f"📊 {best_method.upper()} Method Statistics:")
                self.logger.info(f"   Trials: {len(method_values)}")
                self.logger.info(f"   Mean: {sum(method_values) / len(method_values):.4f}")
                self.logger.info(f"   Best: {max(method_values):.4f}")
                self.logger.info(f"   Std: {(sum((x - sum(method_values)/len(method_values))**2 for x in method_values) / len(method_values))**0.5:.4f}")

        self.logger.info("=" * 80)

        # Call parent logging
        self._log_final_results_to_wandb(study, final_results)


def main():
    """Main entry point for Qwen2.5-Coder-7B HumanEval optimization."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Coder-7B HumanEval Task Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: 2 for Qwen2.5-Coder-7B)",
    )

    # Dataset configuration
    parser.add_argument(
        "--train-limit", type=int, default=None, help="Number of training samples to load (default: 80)"
    )
    parser.add_argument(
        "--contrastive-pairs-limit",
        type=int,
        default=None,
        help="Number of contrastive pairs for steering training (default: 40, bounded by train-limit)",
    )
    parser.add_argument(
        "--val-limit", type=int, default=None, help="Number of validation samples to load (default: 40)"
    )
    parser.add_argument("--test-limit", type=int, default=None, help="Number of test samples to load (default: 80)")

    # Optimization configuration
    parser.add_argument(
        "--study-name", type=str, default=None, help="Optuna study name (default: qwen25_coder_humaneval_optimization)"
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Number of optimization trials (default: 50)")
    parser.add_argument(
        "--n-startup-trials", type=int, default=None, help="Random exploration trials before TPE kicks in (default: 10)"
    )
    parser.add_argument(
        "--layer-range", type=int, nargs=2, default=None, help="Layer search range as two integers (default: 16 24)"
    )

    # WandB configuration
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable WandB experiment tracking (requires 'wandb login' first)"
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")

    # Utility options
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test run (10 trials, 20 train, 15 val samples)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default=None, help="Log output to file (in addition to console)")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Configure logging handlers
    handlers = [logging.StreamHandler()]  # Console output
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))  # File output

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers
    )

    logger = logging.getLogger(__name__)

    # Quick test mode overrides
    if args.quick_test:
        args.n_trials = 10
        args.train_limit = 20
        args.val_limit = 15
        args.test_limit = 20
        logger.info("🚀 Quick test mode enabled")

    # Display configuration
    logger.info("🤖 QWEN2.5-CODER-7B HUMANEVAL OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🔧 CONFIGURATION:")
    logger.info(f"   Model: {args.model_path or get_recommended_config_for_qwen25_coder()['model_name']}")
    logger.info(f"   Batch Size: {args.batch_size or get_recommended_config_for_qwen25_coder()['batch_size']}")
    logger.info(f"   Trials: {args.n_trials or get_recommended_config_for_qwen25_coder()['n_trials']}")
    logger.info(f"   Train/Val/Test: {args.train_limit or 80}/{args.val_limit or 40}/{args.test_limit or 80}")
    logger.info(f"   Datasets: HumanEval (train/val/test) - Python programming problems from OpenAI")
    logger.info(f"   WandB: {'Enabled' if args.use_wandb else 'Disabled'}")

    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} devices)")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Memory warning for Qwen2.5-Coder-7B
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 20:
            logger.warning(f"⚠️  GPU has {vram_gb:.1f}GB VRAM. Qwen2.5-Coder-7B requires ~14GB+. Consider:")
            logger.warning("   - Using smaller batch size (--batch-size 1)")
            logger.warning("   - Setting CUDA_VISIBLE_DEVICES to limit GPU usage")
    else:
        logger.error("❌ No CUDA detected - Qwen2.5-Coder-7B requires GPU!")
        return None

    logger.info("=" * 80)

    # Environment variable check
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        logger.warning("⚠️  HF_ALLOW_CODE_EVAL not set. HumanEval requires code execution.")
        logger.warning("   Set: export HF_ALLOW_CODE_EVAL='1'")
        logger.info("=" * 80)

    # Create configuration and pipeline
    try:
        config = create_qwen25_coder_config(args)
        pipeline = Qwen25CoderHumanEvalPipeline(config)

        # Run optimization
        logger.info("🚀 Starting HumanEval optimization for Qwen2.5-Coder-7B...")
        results = pipeline.run_optimization()

        # Enhanced result display
        pipeline._log_enhanced_results(pipeline._create_optuna_study(), results)

        logger.info("✅ Qwen2.5-Coder-7B HumanEval optimization completed successfully!")
        logger.info(f"📂 Results saved to: {config.output_dir}")
        logger.info(f"🗄️  Study database: {config.db_url}")

        if config.use_wandb:
            logger.info("📊 WandB: Check your WandB dashboard for run details")

        return results

    except KeyboardInterrupt:
        logger.info("🛑 Optimization interrupted by user")
        return None

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Cleanup
        if "pipeline" in locals():
            pipeline.cleanup_memory()


if __name__ == "__main__":
    main()