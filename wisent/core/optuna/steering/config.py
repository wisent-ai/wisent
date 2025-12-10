"""
Configuration dataclass for the steering optimization pipeline.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from wisent.core.steering_methods import SteeringMethodRegistry
from wisent.core.utils.device import resolve_default_device


@dataclass
class OptimizationConfig:
    """Configuration for dataset-agnostic optimization pipeline."""

    model_name: str = "realtreetune/rho-1b-sft-GSM8K"
    device: str = field(default_factory=resolve_default_device)

    # Data source: either task (lm-eval benchmark) OR trait (synthetic pairs)
    train_dataset: str = "gsm8k"
    val_dataset: str = "gsm8k"
    test_dataset: str = "gsm8k"
    
    # Synthetic trait for contrastive pair generation (e.g., "british", "random", "refusal")
    # If set, overrides train_dataset for pair generation
    trait: Optional[str] = None
    trait_label: str = "positive"  # Label for the trait direction
    
    # Evaluator configuration (auto-selected based on task)
    eval_prompts: Optional[str] = None  # Path to custom evaluation prompts JSON
    num_eval_prompts: int = 30  # Number of prompts for refusal/personalization evaluation
    
    # Custom evaluator (for --task custom)
    custom_evaluator: Optional[str] = None  # Module path or file:function
    custom_evaluator_kwargs: Optional[dict] = None  # Kwargs for custom evaluator

    # Training configuration
    train_limit: int = 50  # How many training samples to load
    contrastive_pairs_limit: int = 20  # How many contrastive pairs to extract for steering training

    # Evaluation configuration
    val_limit: int = 50  # How many validation samples to load
    test_limit: int = 100  # How many test samples to load

    layer_search_range: tuple[int, int] = (15, 20)
    probe_type: str = "logistic_regression"  # Fixed probe type
    steering_methods: list[str] = field(default_factory=lambda: SteeringMethodRegistry.list_methods())

    # Optuna study configuration
    study_name: str = "optimization_pipeline"
    db_url: str = field(
        default_factory=lambda: f"sqlite:///{os.path.dirname(os.path.dirname(__file__))}/optuna_studies.db"
    )
    n_trials: int = 50
    n_startup_trials: int = 10  # Random exploration before TPE kicks in
    sampler: str = "TPE"
    pruner: str = "MedianPruner"

    # WandB configuration
    wandb_project: str = "wisent-optimization"
    use_wandb: bool = False

    batch_size: int = 8
    max_length: int = 512
    max_new_tokens: int = 256
    seed: int = 42

    temperature: float = None  # Will use inference config if None
    do_sample: bool = None  # Will use inference config if None

    output_dir: str = "outputs/optimization_pipeline"
    cache_dir: str = "cache/optimization_pipeline"

    max_layers_to_search: int = 6
    early_stopping_patience: int = 10
    
    # Early rejection of low-quality vectors during optimization
    enable_early_rejection: bool = True
    early_rejection_snr_threshold: float = 5.0  # Minimum SNR to continue (very relaxed)
    early_rejection_cv_threshold: float = 0.1   # Minimum CV score to continue (very relaxed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
