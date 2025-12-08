"""
Dataset-Agnostic Optimization Pipeline with Optuna

This script builds a reproducible pipeline that:
1. Trains probes and learns steering vectors on the training split
2. Selects the best layer, probe type, steering method, and hyperparameters on validation split via Optuna
3. Evaluates once on the test split with the single best configuration determined on validation

Key features:
- Optuna-based hyperparameter optimization with pruners
- Activation caching for efficiency
- Configurable datasets for train/val/test splits
- Steering evaluation with model re-forwarding
- Reproducibility bundle generation
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from safetensors.torch import save_file as safetensors_save
from tqdm import tqdm

# Optional WandB integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.optuna.steering import data_utils, metrics
from wisent.core.contrastive_pairs.core.response import Response, PositiveResponse, NegativeResponse
from wisent.core.steering_methods import SteeringMethodRegistry, get_steering_method
from wisent.core.task_interface import get_task
from wisent.core.utils.device import empty_device_cache, preferred_dtype, resolve_default_device, resolve_device
from wisent.core.models.inference_config import get_generate_kwargs
from wisent.core.errors import (
    SteeringMethodUnknownError,
    UnknownTypeError,
    ModelArchitectureUnknownError,
    InvalidLayerIdError,
    MissingParameterError,
    OptimizationError,
)
from wisent.core.models.wisent_model import WisentModel
from wisent.core.contrastive_pairs.diagnostics import run_vector_quality_diagnostics, VectorQualityConfig

logger = logging.getLogger(__name__)


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
    use_wandb: bool = False  # TODO

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


class ActivationCache:
    """Efficient activation caching system with proper cache keys."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _generate_cache_key(
        self, split: str, layer_id: int, tokenization_config: dict[str, Any], prompt_variant: str = "default"
    ) -> str:
        """Generate unique cache key for activations."""
        config_str = json.dumps(tokenization_config, sort_keys=True)
        key_data = f"{split}_{layer_id}_{config_str}_{prompt_variant}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"activations_{cache_key}.pkl"

    def has_cached_activations(
        self, split: str, layer_id: int, tokenization_config: dict[str, Any], prompt_variant: str = "default"
    ) -> bool:
        """Check if activations are cached."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        return self._get_cache_path(cache_key).exists()

    def save_activations(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        split: str,
        layer_id: int,
        tokenization_config: dict[str, Any],
        prompt_variant: str = "default",
    ):
        """Save activations to cache."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            "activations": activations,
            "labels": labels,
            "metadata": {
                "split": split,
                "layer_id": layer_id,
                "tokenization_config": tokenization_config,
                "prompt_variant": prompt_variant,
                "timestamp": datetime.now().isoformat(),
                "shape": activations.shape,
            },
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        self.logger.info(f"Cached activations for {split} layer {layer_id}: {activations.shape}")

    def load_activations(
        self, split: str, layer_id: int, tokenization_config: dict[str, Any], prompt_variant: str = "default"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load activations from cache."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            raise FileNotFoundError(f"No cached activations found for key: {cache_key}")

        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        self.logger.info(f"Loaded cached activations for {split} layer {layer_id}: {cache_data['activations'].shape}")
        return cache_data["activations"], cache_data["labels"]


class OptimizationPipeline:
    """Main optimization pipeline using Optuna for hyperparameter search."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Setup output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        self.cache = ActivationCache(config.cache_dir)

        # Initialize WandB if configured
        self.wandb_run = None
        if config.use_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError(
                    "WandB integration enabled but wandb is not installed. Install with: pip install wandb"
                )
            self._init_wandb()

        self.model = None
        self.tokenizer = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None
        # Store task documents for BigCode evaluation
        self.train_task_docs = None
        self.val_task_docs = None
        self.test_task_docs = None
        self.tokenization_config = {
            "max_length": config.max_length,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }

    @property
    def is_coding_task(self) -> bool:
        """Check if the current task requires code execution evaluation."""
        from ...parameters.task_config import CODING_TASKS
        from ..bigcode_integration import is_bigcode_task

        val_dataset = getattr(self.config, "val_dataset", None)
        if not val_dataset:
            return False

        return val_dataset.lower() in CODING_TASKS or is_bigcode_task(val_dataset)

    def run_optimization(self) -> dict[str, Any]:
        """Run the complete optimization pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING OPTIMIZATION PIPELINE WITH OPTUNA")
        self.logger.info("=" * 80)

        # Create timestamped run directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"📁 Run directory: {self.run_dir}")

        self._setup_experiment()
        
        # Compute baseline accuracy once (no steering) for comparison
        self._compute_baseline_accuracy()
        
        study = self._create_optuna_study()
        study.optimize(self._objective_function, n_trials=self.config.n_trials)
        best_trial = study.best_trial
        
        # Save all trials metrics to central JSON
        self._save_all_trials_metrics(study)
        
        final_results = self._final_evaluation(best_trial)
        self._save_reproducibility_bundle(study, final_results)

        # Log final results to WandB
        self._log_final_results_to_wandb(study, final_results)

        self.logger.info("✅ Optimization completed successfully!")
        return final_results

    def _setup_experiment(self):
        """Setup model, tokenizer, and load datasets."""
        self.logger.info("📊 Setting up experiment...")

        # Load model and tokenizer via WisentModel for consistent settings
        self._wisent_model = WisentModel(model_name=self.config.model_name, device=str(self.device))
        
        self.model = self._wisent_model.hf_model
        self.tokenizer = self._wisent_model.tokenizer

        self.model.eval()  # Set to evaluation mode

        # Set left padding for decoder-only models (same as comprehensive evaluation)
        self.tokenizer.padding_side = "left"

        # Setup evaluator based on config
        self._setup_evaluator()

        # Load data based on mode: trait (synthetic) or task (benchmark)
        if self.config.trait:
            self._setup_synthetic_data()
        else:
            self._setup_task_data()

        # Pre-cache activations for all layers on all splits (only for task mode)
        if not self.config.trait:
            self._precache_activations()

    def _setup_evaluator(self):
        """Setup the evaluator based on --task argument.
        
        Task types:
        - 'refusal' → RefusalEvaluator
        - 'personalization' → PersonalizationEvaluator (requires trait)
        - 'custom' → CustomEvaluator (requires custom_evaluator config)
        - benchmark name → TaskEvaluator
        """
        task = (self.config.train_dataset or "").lower()
        
        # Determine evaluator type from task
        if task == "refusal":
            evaluator_type = "refusal"
        elif task == "personalization":
            if not self.config.trait:
                raise ValueError("--trait is required when --task personalization")
            evaluator_type = "personalization"
        elif task == "custom":
            if not getattr(self.config, 'custom_evaluator', None):
                raise ValueError("--custom-evaluator is required when --task custom")
            evaluator_type = "custom"
        else:
            # Benchmark task
            evaluator_type = "task"
        
        self.evaluator_type = evaluator_type
        self.logger.info(f"📊 Using evaluator: {evaluator_type}")
        
        # Setup evaluator based on type
        if evaluator_type == "refusal":
            self._setup_refusal_evaluator()
        elif evaluator_type == "personalization":
            self._setup_personalization_evaluator()
        elif evaluator_type == "custom":
            self._setup_custom_evaluator()
        # task evaluator uses existing benchmark evaluation logic

    def _setup_refusal_evaluator(self):
        """Setup refusal evaluator for compliance/refusal measurement using UncensorBench."""
        from wisent.core.evaluators.steering_evaluators import RefusalEvaluator, EvaluatorConfig
        config = EvaluatorConfig(
            evaluator_type="refusal",
            eval_prompts_path=self.config.eval_prompts,
            num_eval_prompts=self.config.num_eval_prompts,
        )
        self.steering_evaluator = RefusalEvaluator(config, self.config.model_name)
        self.eval_prompts = self.steering_evaluator.get_prompts()
        self.logger.info(f"📊 Loaded {len(self.eval_prompts)} evaluation prompts for refusal evaluation")

    def _setup_personalization_evaluator(self):
        """Setup personalization evaluator for trait steering."""
        from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator, EvaluatorConfig
        config = EvaluatorConfig(
            evaluator_type="personalization",
            trait=self.config.trait,
            eval_prompts_path=self.config.eval_prompts,
            num_eval_prompts=self.config.num_eval_prompts,
        )
        self.steering_evaluator = PersonalizationEvaluator(
            config, self.config.model_name, wisent_model=self._wisent_model
        )
        self.personalization_test_prompts = self.steering_evaluator.get_prompts()
        self.logger.info(f"📊 Setup personalization evaluator for trait: {self.config.trait}")

    def _setup_custom_evaluator(self):
        """Setup custom evaluator from user-provided module/function."""
        import json
        from wisent.core.evaluators.custom import create_custom_evaluator
        
        # Parse custom evaluator kwargs
        custom_kwargs = {}
        if getattr(self.config, 'custom_evaluator_kwargs', None):
            if isinstance(self.config.custom_evaluator_kwargs, str):
                custom_kwargs = json.loads(self.config.custom_evaluator_kwargs)
            else:
                custom_kwargs = self.config.custom_evaluator_kwargs
        
        # Create the custom evaluator
        self.custom_evaluator = create_custom_evaluator(
            self.config.custom_evaluator, **custom_kwargs
        )
        
        # Get test prompts - use trait-based prompts if trait provided, else generic
        if self.config.trait:
            from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator
            self.custom_test_prompts = PersonalizationEvaluator._generate_test_prompts(
                self.config.num_eval_prompts or 30
            )
        else:
            self.custom_test_prompts = [
                "Tell me about yourself.",
                "What do you think about artificial intelligence?",
                "How would you solve world hunger?",
                "Explain quantum computing in simple terms.",
                "What's the meaning of life?",
            ]
        
        self.logger.info(f"📊 Setup custom evaluator: {self.config.custom_evaluator}")

    def _setup_synthetic_data(self):
        """Setup synthetic contrastive pairs for trait-based optimization."""
        self.logger.info(f"📊 Generating synthetic pairs for trait: {self.config.trait}")
        
        from wisent.core.cli.agent.generate_synthetic_pairs import generate_synthetic_pairs
        
        # Generate synthetic contrastive pairs
        # Time budget of 5 minutes, can generate ~20-50 pairs
        pair_set, report = generate_synthetic_pairs(
            model=self._wisent_model,
            prompt=self.config.trait,
            time_budget=5.0,  # 5 minutes
            num_pairs=self.config.contrastive_pairs_limit,
            verbose=False,
        )
        
        # Convert ContrastivePairSet to list of dicts for compatibility
        self.synthetic_pairs = []
        for pair in pair_set.pairs:
            self.synthetic_pairs.append({
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.model_response,
                "negative_response": pair.negative_response.model_response,
            })
        
        # For synthetic mode, we don't have traditional train/val/test samples
        # Instead, we use the synthetic pairs for training and eval prompts for evaluation
        self.train_samples = self.synthetic_pairs
        self.val_samples = []  # Evaluation uses eval_prompts or generated prompts
        self.test_samples = []
        
        self.train_task_docs = self.train_samples
        self.val_task_docs = self.val_samples
        self.test_task_docs = self.test_samples
        
        self.logger.info(f"📊 Generated {len(self.synthetic_pairs)} synthetic pairs")

    def _generate_trait_test_prompts(self) -> list[str]:
        """Generate test prompts for trait evaluation."""
        # Generic prompts that can be used to test any trait
        base_prompts = [
            "Tell me about yourself.",
            "What do you think about the weather today?",
            "Can you help me write a short story?",
            "What's your opinion on modern technology?",
            "How would you describe a perfect day?",
            "Tell me a joke.",
            "What advice would you give to someone starting a new job?",
            "Describe your favorite food.",
            "What do you think about social media?",
            "How do you handle stress?",
            "What's your favorite book and why?",
            "Describe a memorable experience.",
            "What qualities make a good friend?",
            "How would you spend a million dollars?",
            "What's your view on artificial intelligence?",
            "Tell me about a place you'd like to visit.",
            "What hobbies do you enjoy?",
            "How do you define success?",
            "What's the most important lesson you've learned?",
            "Describe your ideal vacation.",
        ]
        return base_prompts[:self.config.num_eval_prompts]

    def _setup_task_data(self):
        """Setup task-based data for benchmark optimization."""
        # Load datasets
        self.train_samples = data_utils.load_dataset_samples(self.config.train_dataset, self.config.train_limit)
        self.val_samples = data_utils.load_dataset_samples(self.config.val_dataset, self.config.val_limit)
        self.test_samples = data_utils.load_dataset_samples(self.config.test_dataset, self.config.test_limit)

        # Store task documents for BigCode evaluation (coding tasks)
        self.train_task_docs = self.train_samples
        self.val_task_docs = self.val_samples
        self.test_task_docs = self.test_samples

        self.logger.info(
            f"Loaded {len(self.train_samples)} train, {len(self.val_samples)} val, {len(self.test_samples)} test samples"
        )

    def _precache_activations(self):
        """Pre-cache activations for all layers and splits to improve efficiency."""
        self.logger.info("🔄 Pre-caching activations for efficiency...")

        layer_range = range(self.config.layer_search_range[0], self.config.layer_search_range[1] + 1)

        splits_data = [("train", self.train_samples), ("val", self.val_samples), ("test", self.test_samples)]

        for split_name, samples in splits_data:
            for layer_id in layer_range:
                if not self.cache.has_cached_activations(split_name, layer_id, self.tokenization_config):
                    self.logger.info(f"Caching activations for {split_name} split, layer {layer_id}")

                    dataset_name = {
                        "train": self.config.train_dataset,
                        "val": self.config.val_dataset,
                        "test": self.config.test_dataset,
                    }[split_name]

                    activations, labels = self._create_probe_data(samples, layer_id, dataset_name)

                    self.cache.save_activations(activations, labels, split_name, layer_id, self.tokenization_config)
                else:
                    self.logger.info(f"Activations already cached for {split_name} split, layer {layer_id}")

    def _compute_baseline_accuracy(self):
        """Compute baseline accuracy on validation set (no steering) once for all trials."""
        self.logger.info("📊 Computing baseline accuracy (no steering)...")
        
        # Different baseline computation based on evaluator type
        if self.evaluator_type == "refusal":
            self.baseline_accuracy = self._compute_baseline_refusal()
        elif self.evaluator_type == "personalization":
            self.baseline_accuracy = self._compute_baseline_personalization()
        else:
            # Task-based evaluation
            self.baseline_accuracy = self._compute_baseline_task()
        
        self.logger.info(f"📊 Baseline accuracy: {self.baseline_accuracy:.4f}")

    def _compute_baseline_task(self):
        """Compute baseline for task-based evaluation."""
        task = get_task(self.config.val_dataset)
        extractor = task.get_extractor()
        
        questions = []
        ground_truths = []
        
        for sample in self.val_samples:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue
            questions.append(qa_pair["formatted_question"])
            ground_truths.append(qa_pair["correct_answer"])
        
        if not questions:
            self.logger.warning("No valid QA pairs for baseline computation")
            return 0.0
        
        # Generate baseline predictions
        predictions = self._generate_baseline_batched(questions)
        
        # Evaluate
        baseline_metrics = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, self.config.val_dataset
        )
        
        return baseline_metrics.get("accuracy", 0.0)

    def _compute_baseline_refusal(self):
        """Compute baseline compliance rate for refusal evaluation using shared evaluator."""
        if not hasattr(self, 'steering_evaluator'):
            self.logger.warning("No steering evaluator setup for refusal baseline")
            return 0.0
        
        prompts = self.steering_evaluator.get_prompts()
        if not prompts:
            return 0.0
        
        # Generate responses without steering
        responses = self._generate_baseline_batched(prompts)
        
        # Evaluate using shared evaluator
        results = self.steering_evaluator.evaluate_responses(responses)
        return results.get("score", results.get("compliance_rate", 0.0))

    def _compute_baseline_personalization(self):
        """Compute baseline trait score for personalization evaluation using shared evaluator."""
        if not hasattr(self, 'steering_evaluator'):
            self.logger.warning("No steering evaluator setup for personalization baseline")
            return 0.0
        
        prompts = self.steering_evaluator.get_prompts()
        if not prompts:
            return 0.0
        
        # Generate responses without steering - these become the baseline for the evaluator
        responses = self._generate_baseline_batched(prompts)
        
        # Store as baseline in the evaluator
        self.steering_evaluator._baseline_responses = responses
        
        # Baseline score is essentially 0 since responses match themselves
        # The actual evaluation will compare steered responses against this baseline
        return 0.0  # Baseline trait alignment before steering
    def _create_probe_data(
        self, samples: list[dict], layer_id: int, dataset_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create contrastive probe training data for a specific layer."""
        self.logger.info(f"Creating probe data from {len(samples)} samples for {dataset_name} on layer {layer_id}")

        # Get task for the specified dataset
        task = get_task(dataset_name)
        extractor = task.get_extractor()
        self.logger.debug(f"Using task: {task.__class__.__name__}, extractor: {extractor.__class__.__name__}")

        texts = []
        labels = []
        success_count = 0
        fail_count = 0

        for i, sample in enumerate(samples):
            try:
                # Extract QA pair
                contrastive_pair = extractor.extract_contrastive_pair(sample, task)

                # Skip samples where contrastive pair extraction failed
                if not contrastive_pair:
                    self.logger.debug(f"Sample {i + 1}: No contrastive pair extracted from keys: {list(sample.keys())}")
                    fail_count += 1
                    continue

                success_count += 1
                self.logger.debug(f"Sample {i + 1}: Successfully extracted contrastive pair")

            except Exception as e:
                self.logger.error(f"Sample {i + 1}: Exception during contrastive pair extraction: {e}")
                fail_count += 1
                continue

            question = contrastive_pair["question"]
            correct_answer = contrastive_pair["correct_answer"]
            incorrect_answer = contrastive_pair["incorrect_answer"]

            # Log contrastive pair details
            self.logger.debug(f"Contrastive pair - Question: ...{question[-50:]}")
            self.logger.debug(f"Contrastive pair - Correct: {correct_answer}, Incorrect: {incorrect_answer}")

            correct_text = f"{question} {correct_answer}"
            texts.append(correct_text)
            labels.append(1)

            incorrect_text = f"{question} {incorrect_answer}"
            texts.append(incorrect_text)
            labels.append(0)

        self.logger.info(
            f"Probe data creation: {success_count} successful, {fail_count} failed. Generated {len(texts)} texts."
        )

        if len(texts) == 0:
            self.logger.error("No texts generated for activation extraction! All contrastive pair extractions failed.")
            return np.array([]), np.array([])

        activations = data_utils.extract_activations_with_hook(
            self.model, self.tokenizer, texts, layer_id, self.config.batch_size, self.config.max_length, self.device
        )

        return activations, np.array(labels)

    def _create_optuna_study(self) -> optuna.Study:
        """Create Optuna study with SQLite persistence and specified sampler/pruner."""
        self.logger.info("📋 Creating Optuna study with SQLite persistence...")
        self.logger.info(f"Database: {self.config.db_url}")
        self.logger.info(f"Study name: {self.config.study_name}")
        self.logger.info(f"🎲 Warmup: {self.config.n_startup_trials} random trials before TPE sampling")

        # Setup sampler
        if self.config.sampler == "TPE":
            sampler = TPESampler(seed=self.config.seed, n_startup_trials=self.config.n_startup_trials)
        elif self.config.sampler == "Random":
            sampler = optuna.samplers.RandomSampler(seed=self.config.seed)
        else:
            sampler = TPESampler(seed=self.config.seed, n_startup_trials=self.config.n_startup_trials)

        # Setup pruner
        if self.config.pruner == "MedianPruner":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == "SuccessiveHalvingPruner":
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Create study with SQLite storage
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.db_url,
            direction="maximize",  # Maximize validation accuracy
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,  # Continue existing study if it exists
        )

        self.logger.info(f"Study created/loaded with {len(study.trials)} existing trials")

        return study

    def _save_steering_vector_dual_format(self, steering_instance, pt_path: Path, safetensors_path: Path) -> bool:
        """Save steering vector in both .pt and safetensors formats."""
        # Save in original .pt format first (preserves all metadata)
        if not steering_instance.save_steering_vector(str(pt_path)):
            self.logger.warning("Failed to save steering vector - method may not be trained")
            return False

        self.logger.info(f"💾 Saved best steering vector to: {pt_path.name}")

        # Also save in safetensors format for HuggingFace compatibility
        try:
            # Load the .pt file and extract steering vector
            data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "steering_vector" in data:
                # Save just the steering vector in safetensors format
                safetensors_save({"steering_vector": data["steering_vector"]}, str(safetensors_path))
                self.logger.info(f"💾 Also saved as safetensors: {safetensors_path.name}")
                return True
            self.logger.warning("Unexpected .pt file structure, safetensors conversion skipped")
            return True  # .pt save was successful
        except Exception as e:
            self.logger.warning(f"Could not create safetensors version: {e}")
            return True  # .pt save was successful

    def _objective_function(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        try:
            # Sample hyperparameters
            layer_id = trial.suggest_int(
                "layer_id", self.config.layer_search_range[0], self.config.layer_search_range[1]
            )

            # Fixed probe type and regularization
            probe_type = self.config.probe_type  # Always logistic_regression
            probe_c = 1.0  # Default regularization strength

            steering_method = trial.suggest_categorical("steering_method", self.config.steering_methods)

            # Get strength range from centralized registry
            if not SteeringMethodRegistry.validate_method(steering_method):
                raise SteeringMethodUnknownError(method=steering_method)
            
            strength_range = SteeringMethodRegistry.get_strength_range(steering_method)
            steering_alpha = trial.suggest_float("steering_alpha", strength_range[0], strength_range[1])

            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)

            # Don't prune based on probe score - focus optimization on steering parameters

            # Build hyperparameters from registry defaults plus steering alpha
            definition = SteeringMethodRegistry.get(steering_method)
            hyperparams = definition.get_default_params()
            hyperparams["steering_alpha"] = steering_alpha

            # Train steering method with early rejection for bad quality vectors
            steering_method_instance, quality_metrics = self._train_steering_method(
                trial, steering_method, layer_id, hyperparams, 
                early_reject=self.config.enable_early_rejection
            )
            
            # If early rejected due to poor quality, return low score without expensive evaluation
            if steering_method_instance is None:
                self.logger.info(f"Trial {trial.number} early rejected - skipping generation")
                trial.set_user_attr("early_rejected", True)
                if quality_metrics:
                    trial.set_user_attr("quality_metrics", quality_metrics)
                return 0.0

            validation_accuracy = self._evaluate_steering_on_validation(
                steering_method_instance, steering_method, layer_id, hyperparams, trial.number, trial
            )

            trial.report(validation_accuracy, step=1)

            # Store baseline and delta for this trial
            baseline = getattr(self, 'baseline_accuracy', 0.0)
            delta = validation_accuracy - baseline
            trial.set_user_attr("baseline_accuracy", baseline)
            trial.set_user_attr("steering_delta", delta)
            
            # Log to WandB
            log_metrics = {"validation_accuracy": validation_accuracy, "probe_score": probe_score,
                          "baseline_accuracy": baseline, "steering_delta": delta}
            if quality_metrics:
                log_metrics.update({f"quality_{k}": v for k, v in quality_metrics.items() if v is not None})
                trial.set_user_attr("quality_metrics", quality_metrics)
                # Print quality metrics for this trial
                self.logger.info(f"Trial {trial.number} quality metrics: "
                    f"SNR={quality_metrics.get('snr', 0):.2f}, "
                    f"CV={quality_metrics.get('cv_score', 0):.3f}, "
                    f"held_out={quality_metrics.get('held_out_transfer', 0):.3f}, "
                    f"cv_class={quality_metrics.get('cv_classification_accuracy', 0):.3f}, "
                    f"cohens_d={quality_metrics.get('cohens_d', 0):.2f}, "
                    f"quality={quality_metrics.get('overall_quality', 'N/A')}")
            self.logger.info(f"Trial {trial.number} result: steered={validation_accuracy:.4f}, baseline={baseline:.4f}, delta={delta:+.4f}")
            self._log_trial_to_wandb(trial, log_metrics)

            return validation_accuracy

        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return 0.0

    def _train_and_evaluate_probe(self, trial: optuna.Trial, layer_id: int, probe_type: str, probe_c: float) -> float:
        """Train probe on training data and evaluate on validation data using cached activations."""
        # Load cached training activations
        X_train, y_train = self.cache.load_activations("train", layer_id, self.tokenization_config)

        # Train probe
        if probe_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            probe = LogisticRegression(C=probe_c, random_state=self.config.seed, max_iter=1000)
            probe.fit(X_train, y_train)
        else:
            raise UnknownTypeError(entity_type="probe_type", value=probe_type)

        # Evaluate on validation data using cached activations
        X_val, y_val = self.cache.load_activations("val", layer_id, self.tokenization_config)

        from sklearn.metrics import roc_auc_score

        y_pred_proba = probe.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5

        # Don't store the probe object - it can't be JSON serialized
        # The probe will be retrained in the final evaluation if needed

    def _train_steering_method(
        self, trial: optuna.Trial, method_name: str, layer_id: int, hyperparams: dict[str, Any],
        early_reject: bool = True
    ) -> tuple[Any, dict[str, Any] | None]:
        """Train steering method on training data with optional early rejection.
        
        Args:
            trial: Optuna trial
            method_name: Name of steering method
            layer_id: Layer to train on
            hyperparams: Hyperparameters for steering method
            early_reject: If True, run quality diagnostics and reject poor vectors early
            
        Returns:
            Tuple of (steering_instance, quality_metrics) where quality_metrics is None if skipped
        """
        # Use contrastive_pairs_limit with bounds checking
        contrastive_limit = min(self.config.contrastive_pairs_limit, len(self.train_samples))
        
        # For trait mode, use synthetic pairs; for task mode, create from task samples
        if self.config.trait and hasattr(self, 'synthetic_pairs'):
            contrastive_pairs = self._create_synthetic_contrastive_pairs(layer_id, limit=contrastive_limit)
        else:
            contrastive_pairs = self._create_contrastive_pairs(
                self.train_samples, layer_id, self.config.train_dataset, limit=contrastive_limit
            )

        # Early rejection: check quality of contrastive pairs before expensive training
        quality_metrics = None
        if early_reject and len(contrastive_pairs.pairs) >= 5:
            try:
                # Extract activations for quality check
                pos_activations = []
                neg_activations = []
                
                for pair in contrastive_pairs.pairs:
                    if hasattr(pair.positive_response, 'activations') and pair.positive_response.activations is not None:
                        pos_activations.append(pair.positive_response.activations)
                    if hasattr(pair.negative_response, 'activations') and pair.negative_response.activations is not None:
                        neg_activations.append(pair.negative_response.activations)
                
                if len(pos_activations) >= 5 and len(neg_activations) >= 5:
                    pos_stacked = torch.stack(pos_activations).squeeze(1)  # Remove batch dim if present
                    neg_stacked = torch.stack(neg_activations).squeeze(1)
                    
                    # Use configurable thresholds for optimization
                    optimization_quality_config = VectorQualityConfig(
                        snr_critical=self.config.early_rejection_snr_threshold,
                        cv_score_critical=self.config.early_rejection_cv_threshold,
                        convergence_critical=0.3,  # Very relaxed
                    )
                    
                    quality_report, diagnostics_report = run_vector_quality_diagnostics(
                        pos_stacked, neg_stacked, config=optimization_quality_config
                    )
                    
                    quality_metrics = {
                        "snr": quality_report.snr,
                        "cv_score": quality_report.cv_score_mean,
                        "convergence": quality_report.convergence_score,
                        "overall_quality": quality_report.overall_quality,
                        "has_critical_issues": diagnostics_report.has_critical_issues,
                        "held_out_transfer": quality_report.held_out_transfer,
                        "cv_classification_accuracy": quality_report.cv_classification_accuracy,
                        "cohens_d": quality_report.cohens_d,
                        "pca_pc1_variance": quality_report.pca_pc1_variance,
                        "silhouette_score": quality_report.silhouette_score,
                    }
                    
                    # Early reject if quality is critically bad
                    if diagnostics_report.has_critical_issues:
                        self.logger.info(
                            f"Early rejecting trial (layer={layer_id}): SNR={quality_report.snr:.2f}, "
                            f"CV={quality_report.cv_score_mean:.3f}, quality={quality_report.overall_quality}"
                        )
                        return None, quality_metrics
                        
            except Exception as e:
                self.logger.debug(f"Quality check failed, continuing without early rejection: {e}")

        # Use centralized registry to create and train steering method
        if not SteeringMethodRegistry.validate_method(method_name):
            raise SteeringMethodUnknownError(method=method_name)
        
        # Create instance from registry
        steering_instance = get_steering_method(method_name, device=self.device, **hyperparams)

        # Train steering method
        steering_instance.train(contrastive_pairs, layer_id)
        return steering_instance, quality_metrics

    def _create_contrastive_pairs(
        self, samples: list[dict], layer_id: int, dataset_name: str, limit: Optional[int] = None
    ) -> ContrastivePairSet:
        """Create contrastive pairs with activations for steering training."""
        contrastive_pairs = []
        task = get_task(dataset_name)
        extractor = task.get_extractor()

        samples_to_use = samples[:limit] if limit else samples

        for sample in samples_to_use:
            contrastive_pair = extractor.extract_contrastive_pair(sample, task)
            if contrastive_pair:
                # Log contrastive pair details
                self.logger.debug(f"Creating contrastive pair - Question: ...{contrastive_pair['question'][-50:]}")
                self.logger.debug(
                    f"Creating contrastive pair - Correct: {contrastive_pair['correct_answer']}, Incorrect: {contrastive_pair['incorrect_answer']}"
                )

                positive_response = PositiveResponse(model_response=contrastive_pair["correct_answer"], label="1")
                negative_response = NegativeResponse(model_response=contrastive_pair["incorrect_answer"], label="0")

                pair = ContrastivePair(
                    prompt=contrastive_pair["question"],
                    positive_response=positive_response,
                    negative_response=negative_response,
                )
                contrastive_pairs.append(pair)

        pair_set = ContrastivePairSet(name=f"{dataset_name}_training", pairs=contrastive_pairs)

        # Extract activations for all pairs in batches
        if pair_set.pairs:
            all_texts = []
            text_to_pair_mapping = []

            for pair_idx, pair in enumerate(pair_set.pairs):
                pos_text = f"{pair.prompt} {pair.positive_response.model_response}"
                neg_text = f"{pair.prompt} {pair.negative_response.model_response}"

                all_texts.extend([pos_text, neg_text])
                text_to_pair_mapping.extend([(pair_idx, "positive"), (pair_idx, "negative")])

            all_activations = self._extract_batch_activations(all_texts, layer_id)

            for text_idx, (pair_idx, response_type) in enumerate(text_to_pair_mapping):
                activation = all_activations[text_idx]

                if response_type == "positive":
                    pair_set.pairs[pair_idx].positive_response.activations = activation
                else:
                    pair_set.pairs[pair_idx].negative_response.activations = activation

        return pair_set

    def _create_synthetic_contrastive_pairs(
        self, layer_id: int, limit: Optional[int] = None
    ) -> ContrastivePairSet:
        """Create contrastive pairs from pre-generated synthetic pairs for trait mode."""
        if not hasattr(self, 'synthetic_pairs') or not self.synthetic_pairs:
            self.logger.warning("No synthetic pairs available")
            return ContrastivePairSet(name=f"synthetic_{self.config.trait}", pairs=[])
        
        pairs_to_use = self.synthetic_pairs[:limit] if limit else self.synthetic_pairs
        contrastive_pairs = []
        
        for pair_data in pairs_to_use:
            # Synthetic pairs should already have prompt, positive_response, negative_response structure
            if isinstance(pair_data, dict):
                prompt = pair_data.get("prompt", pair_data.get("question", ""))
                positive_text = pair_data.get("positive_response", pair_data.get("positive", ""))
                negative_text = pair_data.get("negative_response", pair_data.get("negative", ""))
            elif hasattr(pair_data, 'prompt'):
                # Already a ContrastivePair object
                contrastive_pairs.append(pair_data)
                continue
            else:
                continue
                
            positive_response = PositiveResponse(model_response=positive_text, label="1")
            negative_response = NegativeResponse(model_response=negative_text, label="0")
            
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=positive_response,
                negative_response=negative_response,
            )
            contrastive_pairs.append(pair)
        
        pair_set = ContrastivePairSet(name=f"synthetic_{self.config.trait}", pairs=contrastive_pairs)
        
        # Extract activations for all pairs in batches
        if pair_set.pairs:
            all_texts = []
            text_to_pair_mapping = []
            
            for pair_idx, pair in enumerate(pair_set.pairs):
                pos_text = f"{pair.prompt} {pair.positive_response.model_response}"
                neg_text = f"{pair.prompt} {pair.negative_response.model_response}"
                
                all_texts.extend([pos_text, neg_text])
                text_to_pair_mapping.extend([(pair_idx, "positive"), (pair_idx, "negative")])
            
            all_activations = self._extract_batch_activations(all_texts, layer_id)
            
            for text_idx, (pair_idx, response_type) in enumerate(text_to_pair_mapping):
                activation = all_activations[text_idx]
                
                if response_type == "positive":
                    pair_set.pairs[pair_idx].positive_response.activations = activation
                else:
                    pair_set.pairs[pair_idx].negative_response.activations = activation
        
        self.logger.info(f"Created {len(pair_set.pairs)} synthetic contrastive pairs for layer {layer_id}")
        return pair_set

    def _extract_batch_activations(self, texts: list[str], layer_id: int) -> list[torch.Tensor]:
        """Extract activations for multiple texts in batches."""
        if not texts:
            return []

        all_activations = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
            ).to(self.device)

            batch_activations = []

            def batch_hook_fn(module, input, output):
                with torch.no_grad():
                    hidden_states = output[0] if isinstance(output, tuple) else output

                    last_token_acts = hidden_states[:, -1, :].detach().clone()
                    batch_activations.append(last_token_acts)

            if hasattr(self.model, "transformer"):
                target_layer = self.model.transformer.h[layer_id]
            elif hasattr(self.model, "model"):
                target_layer = self.model.model.layers[layer_id]
            else:
                raise ModelArchitectureUnknownError()

            handle = target_layer.register_forward_hook(batch_hook_fn)

            try:
                with torch.no_grad():
                    _ = self.model(**inputs)
            finally:
                handle.remove()
                empty_device_cache(self.device.type)

            if batch_activations:
                batch_tensor = batch_activations[0]
                for j in range(batch_tensor.shape[0]):
                    all_activations.append(batch_tensor[j].unsqueeze(0))

        return all_activations

    def _extract_single_activation(self, text: str, layer_id: int) -> torch.Tensor:
        """Extract activation for a single text."""
        activations = self._extract_batch_activations([text], layer_id)
        return activations[0] if activations else torch.zeros(1, self.model.config.hidden_size, device=self.device)

    def _evaluate_steering_on_validation(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
        trial_number: int = 0,
        trial=None,
    ) -> float:
        """Evaluate steering method on validation data by re-running forward passes."""
        if steering_instance is None:
            return 0.0

        # Dispatch to appropriate evaluator
        if self.evaluator_type == "refusal":
            return self._evaluate_refusal(steering_instance, method_name, layer_id, hyperparams)
        elif self.evaluator_type == "personalization":
            return self._evaluate_personalization(steering_instance, method_name, layer_id, hyperparams)
        else:
            # Task-based evaluation (default)
            return self._evaluate_task(steering_instance, method_name, layer_id, hyperparams, trial_number, trial)

    def _evaluate_task(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
        trial_number: int = 0,
        trial=None,
    ) -> float:
        """Task-based evaluation using benchmark metrics."""
        # Generate predictions with steering applied
        predictions = []
        ground_truths = []
        task_docs = []  # Preserve original task documents for BigCode evaluation

        task = get_task(self.config.val_dataset)
        extractor = task.get_extractor()

        # Collect all questions for batched processing (use ALL validation samples)
        questions = []
        ground_truths = []
        valid_samples = []  # Keep track of samples that produce valid QA pairs

        for sample in tqdm(
            self.val_samples, desc="Extracting validation QA pairs", leave=False
        ):  # Use all validation samples for reliable evaluation
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue

            question = qa_pair["formatted_question"]
            ground_truth = qa_pair["correct_answer"]
            questions.append(question)
            ground_truths.append(ground_truth)
            valid_samples.append(sample)  # Store the original sample

        # Generate predictions using batched approach
        if questions:
            if steering_instance is None:
                predictions = self._generate_baseline_batched(questions)
            else:
                # Extract the appropriate strength parameter based on method
                # CAA uses steering_alpha directly
                strength = hyperparams["steering_alpha"]

                predictions = self._generate_with_steering_batched(steering_instance, questions, strength, layer_id)

            # Log sample predictions for debugging
            for i, (pred, gt) in enumerate(zip(predictions[:3], ground_truths[:3])):
                self.logger.debug(f"{method_name.upper()} Sample {i} - Model: ...{pred[-50:] if pred else 'None'}")
                self.logger.debug(f"{method_name.upper()} Sample {i} - Ground truth: {gt}")
        else:
            predictions = []

        if not predictions:
            return 0.0

        # Save detailed validation results to JSON
        self._save_detailed_validation_results(
            questions,
            ground_truths,
            predictions,
            trial_number,
            trial=trial,
            steering_method=method_name,
            layer_id=layer_id,
            hyperparams=hyperparams,
        )

        # Prepare task docs for BigCode evaluation (if coding task)
        task_docs = valid_samples[: len(predictions)] if valid_samples else []

        # Evaluate benchmark performance (with task docs for coding tasks)
        benchmark_metrics = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, self.config.val_dataset, task_docs=task_docs
        )

        return benchmark_metrics.get("accuracy", 0.0)

    def _evaluate_refusal(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
    ) -> float:
        """Evaluate steering using shared refusal evaluator."""
        if not hasattr(self, 'steering_evaluator'):
            self.logger.warning("No steering evaluator setup for refusal evaluation")
            return 0.0
        
        prompts = self.steering_evaluator.get_prompts()
        if not prompts:
            return 0.0
        
        # Generate steered responses
        strength = hyperparams["steering_alpha"]
        responses = self._generate_with_steering_batched(steering_instance, prompts, strength, layer_id)
        
        # Evaluate using shared evaluator
        results = self.steering_evaluator.evaluate_responses(responses)
        return results.get("score", results.get("compliance_rate", 0.0))

    def _evaluate_personalization(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
    ) -> float:
        """Evaluate steering using shared personalization evaluator."""
        if not hasattr(self, 'steering_evaluator'):
            self.logger.warning("No steering evaluator setup for personalization evaluation")
            return 0.0
        
        prompts = self.steering_evaluator.get_prompts()
        if not prompts:
            return 0.0
        
        # Generate steered responses
        strength = hyperparams["steering_alpha"]
        responses = self._generate_with_steering_batched(steering_instance, prompts, strength, layer_id)
        
        # Evaluate using shared evaluator
        results = self.steering_evaluator.evaluate_responses(responses)
        return results.get("score", 0.0)

    def _generate_with_caa_steering(self, caa, question: str, alpha: float, layer_id: int) -> str:
        """Generate response with CAA steering applied."""
        if not hasattr(caa, "steering_vector") or caa.steering_vector is None:
            return self._generate_baseline(question)

        return self._generate_with_steering_hook(question, caa.steering_vector, layer_id, alpha)

    def _generate_with_steering_hook(
        self, question: str, steering_vector: torch.Tensor, layer_id: int, alpha: float
    ) -> str:
        """Generate response with steering vector applied via hook (re-runs forward pass)."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)

        def steering_hook(module, input, output):
            """Hook that applies steering vector during forward pass."""
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Apply steering to the last token
                hidden_states[:, -1, :] += alpha * steering_vector.to(hidden_states.device)
                return (hidden_states, *output[1:])
            hidden_states = output
            hidden_states[:, -1, :] += alpha * steering_vector.to(hidden_states.device)
            return hidden_states

        # Register hook on target layer
        if hasattr(self.model, "transformer"):
            target_layer = self.model.transformer.h[layer_id]
        elif hasattr(self.model, "model"):
            target_layer = self.model.model.layers[layer_id]
        else:
            raise ModelArchitectureUnknownError()

        handle = target_layer.register_forward_hook(steering_hook)

        try:
            # Get inference config settings with optional overrides from self.config
            gen_kwargs = get_generate_kwargs(max_new_tokens=self.config.max_new_tokens)
            if self.config.do_sample is not None:
                gen_kwargs["do_sample"] = self.config.do_sample
            if self.config.temperature is not None:
                gen_kwargs["temperature"] = self.config.temperature

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return response.strip()

    def _generate_baseline(self, question: str) -> str:
        """Generate baseline response without steering."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)

        # Get inference config settings with optional overrides from self.config
        gen_kwargs = get_generate_kwargs(max_new_tokens=self.config.max_new_tokens)
        if self.config.do_sample is not None:
            gen_kwargs["do_sample"] = self.config.do_sample
        if self.config.temperature is not None:
            gen_kwargs["temperature"] = self.config.temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return response.strip()

    def _generate_baseline_batched(self, questions: list[str]) -> list[str]:  # TODO
        """Generate baseline responses in batches without steering."""
        if not questions:
            return []

        batch_size = self.config.batch_size
        all_responses = []

        # Get inference config settings with optional overrides from self.config
        gen_kwargs = get_generate_kwargs(max_new_tokens=self.config.max_new_tokens)
        if self.config.do_sample is not None:
            gen_kwargs["do_sample"] = self.config.do_sample
        if self.config.temperature is not None:
            gen_kwargs["temperature"] = self.config.temperature

        # Process questions in batches
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating baseline predictions", leave=False):
            batch_questions = questions[i : i + batch_size]

            # Batch tokenization with padding
            inputs = self.tokenizer(
                batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode responses
            batch_responses = []
            for j, output in enumerate(outputs):
                input_length = inputs.input_ids[j].shape[0]
                response = self.tokenizer.decode(output[input_length:], skip_special_tokens=True)
                batch_responses.append(response.strip())

            all_responses.extend(batch_responses)

        return all_responses

    def _generate_with_steering_batched(
        self, steering_instance: Any, questions: list[str], alpha: float, layer_id: int
    ) -> list[str]:
        """Generate responses with steering applied in batches using apply_steering()."""
        if not questions:
            return []

        batch_size = self.config.batch_size
        all_responses = []

        # Get inference config settings with optional overrides from self.config
        gen_kwargs = get_generate_kwargs(max_new_tokens=self.config.max_new_tokens)
        if self.config.do_sample is not None:
            gen_kwargs["do_sample"] = self.config.do_sample
        if self.config.temperature is not None:
            gen_kwargs["temperature"] = self.config.temperature

        # Process questions in batches
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating steered predictions", leave=False):
            batch_questions = questions[i : i + batch_size]

            # Batch tokenization with padding
            inputs = self.tokenizer(
                batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
            ).to(self.device)

            def steering_hook(module, input, output):
                """Hook that applies steering using the steering method's apply_steering()."""
                hidden_states = output[0] if isinstance(output, tuple) else output

                # Apply steering using the method's apply_steering() function
                steered = steering_instance.apply_steering(hidden_states, strength=alpha)

                if isinstance(output, tuple):
                    return (steered, *output[1:])
                return steered

            # Register hook on target layer
            if hasattr(self.model, "transformer"):
                if layer_id >= len(self.model.transformer.h):
                    raise InvalidLayerIdError(layer_id=layer_id, num_layers=len(self.model.transformer.h))
                target_layer = self.model.transformer.h[layer_id]
            elif hasattr(self.model, "model"):
                if layer_id >= len(self.model.model.layers):
                    raise InvalidLayerIdError(layer_id=layer_id, num_layers=len(self.model.model.layers))
                target_layer = self.model.model.layers[layer_id]
            else:
                raise ModelArchitectureUnknownError()

            handle = target_layer.register_forward_hook(steering_hook)

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **gen_kwargs,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode responses
                batch_responses = []
                for j, output in enumerate(outputs):
                    input_length = inputs.input_ids[j].shape[0]
                    response = self.tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    batch_responses.append(response.strip())

                all_responses.extend(batch_responses)

            finally:
                # Always remove the hook
                handle.remove()

        return all_responses

    def _final_evaluation(self, best_trial: optuna.Trial) -> dict[str, Any]:
        """Run final evaluation on test split with best configuration."""
        self.logger.info("🏆 Running final evaluation with best configuration...")

        # Extract best hyperparameters
        # Handle both real trials and FixedTrial objects
        if hasattr(best_trial, "params") and best_trial.params:
            best_params = best_trial.params
        elif hasattr(best_trial, "_params"):
            best_params = best_trial._params
        else:
            # Fallback - this shouldn't happen
            raise MissingParameterError(params=["trial.params"])
        layer_id = best_params["layer_id"]

        self.logger.info(f"Best configuration: {best_params}")

        # Re-train best probe and steering method on training data
        from sklearn.linear_model import LogisticRegression

        # Train best probe with fixed probe_c
        X_train, y_train = self.cache.load_activations("train", layer_id, self.tokenization_config)
        probe = LogisticRegression(C=1.0, random_state=self.config.seed, max_iter=1000)  # Fixed probe_c
        probe.fit(X_train, y_train)

        # Train best steering method (no early rejection for final evaluation)
        steering_method = best_params.get("steering_method", "caa")  # Default to CAA if missing
        steering_instance, _ = self._train_steering_method(
            best_trial, steering_method, layer_id, best_params, early_reject=False
        )

        # Save the best steering vector in both formats
        if steering_instance and hasattr(steering_instance, "save_steering_vector"):
            pt_path = self.run_dir / "best_steering_vector.pt"
            safetensors_path = self.run_dir / "best_steering_vector.safetensors"
            self._save_steering_vector_dual_format(steering_instance, pt_path, safetensors_path)

        # Generate baseline predictions (no steering)
        self.logger.info("Generating baseline predictions...")
        baseline_predictions, test_ground_truths, test_questions, test_task_docs = self._generate_test_predictions(
            None, None, layer_id, 0.0
        )

        # Generate steered predictions
        self.logger.info("Generating steered predictions...")

        # Extract the appropriate strength parameter based on method and available parameters
        method_name = best_params.get("steering_method", "caa")  # Default to CAA if missing
        # CAA uses steering_alpha
        strength = best_params["steering_alpha"]

        steered_predictions, _, _, _ = self._generate_test_predictions(
            steering_instance, method_name, layer_id, strength
        )

        # Save detailed test results to JSON
        if test_questions and test_ground_truths and baseline_predictions and steered_predictions:
            self._save_detailed_test_results(
                test_questions,
                test_ground_truths,
                baseline_predictions,
                steered_predictions,
                best_trial=best_trial,
                best_params=best_params,
                layer_id=layer_id,
                steering_method=method_name,
            )

        # Calculate benchmark metrics (with real task docs for coding tasks)
        baseline_benchmark_metrics = metrics.evaluate_benchmark_performance(
            baseline_predictions, test_ground_truths, self.config.test_dataset, task_docs=test_task_docs
        )
        steered_benchmark_metrics = metrics.evaluate_benchmark_performance(
            steered_predictions, test_ground_truths, self.config.test_dataset, task_docs=test_task_docs
        )

        # Evaluate probe on test data
        X_test, y_test = self.cache.load_activations("test", layer_id, self.tokenization_config)
        test_probe_metrics = self._evaluate_probe_metrics(probe, X_test, y_test)

        # Calculate improvement
        accuracy_improvement = steered_benchmark_metrics.get("accuracy", 0.0) - baseline_benchmark_metrics.get(
            "accuracy", 0.0
        )

        final_results = {
            "best_trial_params": best_params,
            "best_validation_score": getattr(best_trial, "value", None),
            "baseline_benchmark_metrics": baseline_benchmark_metrics,
            "steered_benchmark_metrics": steered_benchmark_metrics,
            "accuracy_improvement": accuracy_improvement,
            "test_probe_metrics": test_probe_metrics,
            "config": self.config.to_dict(),
            "num_test_samples": len(test_ground_truths),
        }

        # Log final results
        self.logger.info("=" * 60)
        self.logger.info("🏆 FINAL TEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Baseline accuracy: {baseline_benchmark_metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"Steered accuracy: {steered_benchmark_metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"Improvement: {accuracy_improvement:+.4f}")
        self.logger.info(f"Probe AUC: {test_probe_metrics.get('auc', 0.5):.4f}")
        self.logger.info(f"Test samples: {len(test_ground_truths)}")
        
        # Log quality metrics from best trial if available
        quality_metrics = best_trial.user_attrs.get("quality_metrics") if hasattr(best_trial, "user_attrs") else None
        if quality_metrics:
            self.logger.info("-" * 60)
            self.logger.info("📊 VECTOR QUALITY METRICS")
            self.logger.info("-" * 60)
            self.logger.info(f"Overall quality: {quality_metrics.get('overall_quality', 'N/A')}")
            if quality_metrics.get('snr') is not None:
                self.logger.info(f"Signal-to-noise ratio: {quality_metrics['snr']:.2f}")
            if quality_metrics.get('cv_score') is not None:
                self.logger.info(f"Cross-validation score: {quality_metrics['cv_score']:.3f}")
            if quality_metrics.get('convergence') is not None:
                self.logger.info(f"Convergence: {quality_metrics['convergence']:.3f}")
            if quality_metrics.get('held_out_transfer') is not None:
                self.logger.info(f"Held-out transfer: {quality_metrics['held_out_transfer']:.3f}")
            if quality_metrics.get('cv_classification_accuracy') is not None:
                self.logger.info(f"CV classification accuracy: {quality_metrics['cv_classification_accuracy']:.3f}")
            if quality_metrics.get('cohens_d') is not None:
                self.logger.info(f"Cohen's d: {quality_metrics['cohens_d']:.2f}")
            if quality_metrics.get('pca_pc1_variance') is not None:
                self.logger.info(f"PCA PC1 variance: {quality_metrics['pca_pc1_variance']*100:.1f}%")
            if quality_metrics.get('silhouette_score') is not None:
                self.logger.info(f"Silhouette score: {quality_metrics['silhouette_score']:.3f}")
            final_results["quality_metrics"] = quality_metrics
        self.logger.info("=" * 60)

        return final_results

    def _generate_test_predictions(
        self, steering_instance: Any, method_name: str, layer_id: int, alpha: float
    ) -> tuple[list[str], list[str], list[str], list[dict]]:
        """Generate predictions on test data using batched generation."""
        # Collect all questions and ground truths for batching
        questions = []
        ground_truths = []
        valid_samples = []  # Keep track of samples that produce valid QA pairs

        task = get_task(self.config.test_dataset)
        extractor = task.get_extractor()

        for sample in self.test_samples:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue

            question = qa_pair["formatted_question"]
            ground_truth = qa_pair["correct_answer"]
            questions.append(question)
            ground_truths.append(ground_truth)
            valid_samples.append(sample)  # Store the original sample

        # Process all questions with appropriate batched method
        if questions:
            try:
                if steering_instance is None:
                    # Baseline generation - use batched method
                    predictions = self._generate_baseline_batched(questions)
                else:
                    # Use unified batched generation with apply_steering()
                    predictions = self._generate_with_steering_batched(steering_instance, questions, alpha, layer_id)

                # Log sample predictions for debugging
                for i, (pred, gt) in enumerate(zip(predictions[:3], ground_truths[:3])):
                    self.logger.debug(f"Test Sample {i} - Model: ...{pred[-50:] if pred else 'None'}")
                    self.logger.debug(f"Test Sample {i} - Ground truth: {gt}")

            except Exception as e:
                self.logger.warning(f"Batched generation failed for test: {e}")
                predictions = ["Error"] * len(questions)
        else:
            predictions = []

        return predictions, ground_truths, questions, valid_samples

    def _evaluate_probe_metrics(self, probe, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate probe metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        y_pred = probe.predict(X_test)
        y_pred_proba = probe.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
        }

    def _create_experiment_metadata(
        self, trial=None, steering_method: str = None, layer_id: int = None, hyperparams: dict = None
    ):
        """Create comprehensive experiment metadata for detailed results."""
        import platform
        from datetime import datetime

        metadata = {
            "trial_info": {
                "trial_number": trial.number if trial else None,
                "trial_params": dict(trial.params) if trial else {},
                "trial_state": str(getattr(trial, "state", "RUNNING")) if trial else None,
            },
            "model_config": {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "is_coding_task": self.is_coding_task,
            },
            "dataset_config": {
                "train_dataset": self.config.train_dataset,
                "val_dataset": self.config.val_dataset,
                "test_dataset": self.config.test_dataset,
                "train_limit": self.config.train_limit,
                "val_limit": self.config.val_limit,
                "test_limit": self.config.test_limit,
                "contrastive_pairs_limit": self.config.contrastive_pairs_limit,
            },
            "steering_config": {
                "steering_method": steering_method,
                "layer_id": layer_id,
                "hyperparams": hyperparams or {},
                "layer_search_range": self.config.layer_search_range,
                "probe_type": self.config.probe_type,
                "available_steering_methods": self.config.steering_methods,
            },
            "optimization_config": {
                "study_name": self.config.study_name,
                "sampler": self.config.sampler,
                "pruner": self.config.pruner,
                "n_trials": self.config.n_trials,
                "n_startup_trials": self.config.n_startup_trials,
            },
            "generation_config": {
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": self.config.do_sample,
            },
            "run_info": {
                "timestamp": datetime.now().isoformat(),
                "run_dir": str(self.run_dir),
                "output_dir": self.config.output_dir,
                "cache_dir": self.config.cache_dir,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "wandb_config": {
                "use_wandb": self.config.use_wandb,
                "wandb_project": self.config.wandb_project,
            }
            if hasattr(self.config, "use_wandb")
            else {},
        }

        return metadata

    def _save_detailed_validation_results(
        self,
        questions: list[str],
        ground_truths: list[str],
        predictions: list[str],
        trial_number: int,
        trial=None,
        steering_method: str = None,
        layer_id: int = None,
        hyperparams: dict = None,
    ):
        """Save detailed validation results to JSON file with experiment metadata."""
        detailed_results = []

        # For coding tasks, use the same BigCode evaluation as accuracy calculation
        if self.is_coding_task:
            # Use evaluate_benchmark_performance to get consistent BigCode evaluation
            eval_results = metrics.evaluate_benchmark_performance(
                predictions, ground_truths, task_name=self.config.val_dataset, task_docs=self.val_task_docs
            )

            # Extract individual correctness from evaluation details
            eval_details = eval_results.get("evaluation_details", [])

            for i, (question, correct_answer, model_answer) in enumerate(zip(questions, ground_truths, predictions)):
                # Get correctness from BigCode evaluation if available
                is_correct = eval_details[i]["is_correct"] if i < len(eval_details) else False

                detailed_results.append(
                    {
                        "row": i,
                        "question": question,
                        "correct_answer": correct_answer,
                        "model_answer": model_answer,
                        "is_correct": is_correct,
                        "evaluation_method": eval_results.get("evaluation_method", "unknown"),
                        "extracted_code": eval_details[i].get("prediction", model_answer)
                        if i < len(eval_details)
                        else model_answer,
                        "execution_error": eval_details[i].get("execution_error") if i < len(eval_details) else None,
                    }
                )
        else:
            # For non-coding tasks, process each result
            for i, (question, correct_answer, model_answer) in enumerate(zip(questions, ground_truths, predictions)):
                # Use standard evaluation via metrics module
                is_correct = metrics.evaluate_response_correctness(
                    model_answer, correct_answer, self.config.val_dataset
                )

                result_entry = {
                    "row": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "evaluation_method": "string_comparison",
                }

                # Add MC-specific fields if this is a multiple choice task
                if self._should_use_multiple_choice_evaluation():
                    # Extract MC diagnostics directly without custom evaluation
                    import re

                    # Extract available answers from question (A. choice, B. choice, etc.)
                    available_answers = []
                    choice_pattern = r"([A-E])\.\s+(.+?)(?=\n[A-E]\.|$)"
                    matches = re.findall(choice_pattern, question, re.MULTILINE | re.DOTALL)
                    for letter, choice_text in matches:
                        available_answers.append(f"{letter}. {choice_text.strip()}")

                    # Extract model's selected letter from model answer
                    model_selected_letter = "?"
                    model_letter_match = re.search(r"\b([A-E])\b", model_answer.upper())
                    if model_letter_match:
                        model_selected_letter = model_letter_match.group(1)

                    result_entry["available_answers"] = available_answers
                    result_entry["correct_choice_letter"] = correct_answer
                    result_entry["model_selected_letter"] = model_selected_letter

                detailed_results.append(result_entry)

        # Create experiment metadata
        experiment_metadata = self._create_experiment_metadata(trial, steering_method, layer_id, hyperparams)

        # Create final results structure with metadata
        final_results = {"experiment_metadata": experiment_metadata, "results": detailed_results}

        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_detailed_results_trial_{trial_number:03d}_{timestamp}.json"
        filepath = self.run_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Saved detailed validation results to: {filename}")

    def _save_detailed_test_results(
        self,
        questions: list[str],
        ground_truths: list[str],
        baseline_predictions: list[str],
        steered_predictions: list[str],
        best_trial=None,
        best_params: dict = None,
        layer_id: int = None,
        steering_method: str = None,
    ):
        """Save detailed test results to JSON file with both baseline and steered answers and experiment metadata."""
        detailed_results = []

        # For coding tasks, use BigCode evaluation consistently
        if self.is_coding_task:
            # Evaluate baseline predictions with BigCode
            baseline_eval_results = metrics.evaluate_benchmark_performance(
                baseline_predictions, ground_truths, task_name=self.config.test_dataset, task_docs=self.test_task_docs
            )

            # Evaluate steered predictions with BigCode
            steered_eval_results = metrics.evaluate_benchmark_performance(
                steered_predictions, ground_truths, task_name=self.config.test_dataset, task_docs=self.test_task_docs
            )

            baseline_details = baseline_eval_results.get("evaluation_details", [])
            steered_details = steered_eval_results.get("evaluation_details", [])

            for i, (question, correct_answer, baseline_answer, steered_answer) in enumerate(
                zip(questions, ground_truths, baseline_predictions, steered_predictions)
            ):
                # Get correctness from BigCode evaluation
                is_baseline_correct = baseline_details[i]["is_correct"] if i < len(baseline_details) else False
                is_correct = steered_details[i]["is_correct"] if i < len(steered_details) else False

                detailed_results.append(
                    {
                        "row": i,
                        "question": question,
                        "correct_answer": correct_answer,
                        "baseline_model_answer": baseline_answer,
                        "model_answer": steered_answer,
                        "is_baseline_correct": is_baseline_correct,
                        "is_correct": is_correct,
                        "evaluation_method": steered_eval_results.get("evaluation_method", "bigcode_execution"),
                        "baseline_extracted_code": baseline_details[i].get("prediction", baseline_answer)
                        if i < len(baseline_details)
                        else baseline_answer,
                        "steered_extracted_code": steered_details[i].get("prediction", steered_answer)
                        if i < len(steered_details)
                        else steered_answer,
                        "baseline_execution_error": baseline_details[i].get("execution_error")
                        if i < len(baseline_details)
                        else None,
                        "steered_execution_error": steered_details[i].get("execution_error")
                        if i < len(steered_details)
                        else None,
                    }
                )
        else:
            # For non-coding tasks, process each result
            for i, (question, correct_answer, baseline_answer, steered_answer) in enumerate(
                zip(questions, ground_truths, baseline_predictions, steered_predictions)
            ):
                # Use standard evaluation for both baseline and steered answers
                is_baseline_correct = metrics.evaluate_response_correctness(
                    baseline_answer, correct_answer, self.config.test_dataset
                )
                is_correct = metrics.evaluate_response_correctness(
                    steered_answer, correct_answer, self.config.test_dataset
                )

                result_entry = {
                    "row": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "baseline_model_answer": baseline_answer,
                    "model_answer": steered_answer,
                    "is_baseline_correct": is_baseline_correct,
                    "is_correct": is_correct,
                    "evaluation_method": "string_comparison",
                }

                # Add MC-specific fields if this is a multiple choice task
                if self._should_use_multiple_choice_evaluation():
                    # Extract MC diagnostics directly without custom evaluation
                    import re

                    # Extract available answers from question (A. choice, B. choice, etc.)
                    available_answers = []
                    choice_pattern = r"([A-E])\.\s+(.+?)(?=\n[A-E]\.|$)"
                    matches = re.findall(choice_pattern, question, re.MULTILINE | re.DOTALL)
                    for letter, choice_text in matches:
                        available_answers.append(f"{letter}. {choice_text.strip()}")

                    # Extract steered model's selected letter
                    steered_selected_letter = "?"
                    steered_letter_match = re.search(r"\b([A-E])\b", steered_answer.upper())
                    if steered_letter_match:
                        steered_selected_letter = steered_letter_match.group(1)

                    # Extract baseline model's selected letter
                    baseline_selected_letter = "?"
                    baseline_letter_match = re.search(r"\b([A-E])\b", baseline_answer.upper())
                    if baseline_letter_match:
                        baseline_selected_letter = baseline_letter_match.group(1)

                    result_entry["available_answers"] = available_answers
                    result_entry["correct_choice_letter"] = correct_answer
                    result_entry["model_selected_letter"] = steered_selected_letter
                    result_entry["baseline_model_selected_letter"] = baseline_selected_letter

                detailed_results.append(result_entry)

        # Create experiment metadata for test results
        experiment_metadata = self._create_experiment_metadata(
            trial=best_trial,
            steering_method=steering_method or best_params.get("steering_method") if best_params else None,
            layer_id=layer_id,
            hyperparams=best_params,
        )

        # Create final results structure with metadata
        final_results = {"experiment_metadata": experiment_metadata, "results": detailed_results}

        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_detailed_results_{timestamp}.json"
        filepath = self.run_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Saved detailed test results to: {filename}")
        return filename

    def _save_all_trials_metrics(self, study: optuna.Study):
        """Save all trials' quality metrics and scores to a central JSON file."""
        baseline = getattr(self, 'baseline_accuracy', 0.0)
        trials_data = {
            "model": self.config.model_name,
            "task": self.config.val_dataset,
            "timestamp": self.run_timestamp,
            "baseline_accuracy": baseline,
            "n_trials": len(study.trials),
            "trials": []
        }
        
        for trial in study.trials:
            trial_entry = {
                "trial_number": trial.number,
                "validation_accuracy": trial.value,
                "baseline_accuracy": trial.user_attrs.get("baseline_accuracy", baseline),
                "steering_delta": trial.user_attrs.get("steering_delta", (trial.value - baseline) if trial.value else None),
                "state": str(trial.state),
                "params": trial.params,
                "quality_metrics": trial.user_attrs.get("quality_metrics"),
                "early_rejected": trial.user_attrs.get("early_rejected", False),
            }
            trials_data["trials"].append(trial_entry)
        
        # Save to central JSON
        metrics_path = self.run_dir / f"all_trials_metrics_{self.run_timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(trials_data, f, indent=2)
        
        self.logger.info(f"💾 Saved all trials metrics to: {metrics_path}")

    def _save_reproducibility_bundle(self, study: optuna.Study, final_results: dict[str, Any]):
        """Save complete reproducibility bundle."""

        # Save Optuna study
        study_path = self.run_dir / f"optuna_study_{self.run_timestamp}.db"
        study.study_name = str(study_path)

        # Save configuration
        config_path = self.run_dir / f"config_{self.run_timestamp}.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save final results
        results_path = self.run_dir / f"final_results_{self.run_timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        # Save best configuration
        best_config = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "model_name": self.config.model_name,
            "random_seed": self.config.seed,
            "commit_hash": self._get_git_commit_hash(),
            "timestamp": self.run_timestamp,
        }

        best_config_path = self.run_dir / f"best_configuration_{self.run_timestamp}.json"
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)

        # Save study trials summary
        trials_df = study.trials_dataframe()
        trials_path = self.run_dir / f"study_trials_{self.run_timestamp}.csv"
        trials_df.to_csv(trials_path, index=False)

        self.logger.info(f"💾 Reproducibility bundle saved to: {self.run_dir}")
        self.logger.info(f"📊 Study database: {study_path}")
        self.logger.info(f"⚙️ Configuration: {config_path}")
        self.logger.info(f"🏆 Results: {results_path}")
        self.logger.info(f"🎯 Best config: {best_config_path}")

        # Log steering vector if it exists (prefer safetensors format)
        safetensors_path = self.run_dir / "best_steering_vector.safetensors"
        pt_path = self.run_dir / "best_steering_vector.pt"

        if safetensors_path.exists():
            self.logger.info(f"🧭 Steering vector: {safetensors_path.name}")
        elif pt_path.exists():
            self.logger.info(f"🧭 Steering vector: {pt_path.name}")

    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash for reproducibility."""
        try:
            import subprocess

            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None

    def evaluate_only(self, best_params: dict[str, Any]) -> dict[str, Any]:
        """Run evaluation only with provided parameters.

        Args:
            best_params: Dictionary of hyperparameters to use for evaluation

        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("🔬 Running evaluation-only mode with provided parameters")
        self.logger.info(f"Parameters: {best_params}")

        # Setup experiment if not already done
        if self.model is None:
            self._setup_experiment()

        # Create timestamped run directory for evaluation-only mode
        if not hasattr(self, "run_dir"):
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / f"evaluate_only_{self.run_timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Evaluation directory: {self.run_dir}")

        # Create a complete mock trial with all expected parameters
        from optuna.trial import FixedTrial

        # Ensure we have all required parameters for _final_evaluation
        complete_params = {
            "layer_id": best_params.get("layer_id", 15),
            "probe_type": best_params.get("probe_type", "logistic_regression"),
            "probe_c": best_params.get("probe_c", 1.0),
            "steering_method": best_params.get("steering_method", "caa"),
            "steering_alpha": best_params.get("steering_alpha", 0.5),
        }

        fixed_trial = FixedTrial(complete_params)

        # Fix FixedTrial params access issue
        if not hasattr(fixed_trial, "params"):
            fixed_trial.params = complete_params

        # Run final evaluation
        return self._final_evaluation(fixed_trial)

    @classmethod
    def from_saved_study(
        cls, study_path: str, config_path: Optional[str] = None, override_config: Optional[dict[str, Any]] = None
    ):
        """Create pipeline from saved study and optionally saved config.

        Args:
            study_path: Path to the SQLite study database
            config_path: Optional path to saved configuration JSON
            override_config: Optional dict of config values to override

        Returns:
            Tuple of (pipeline, study) ready for evaluation
        """
        # Load config if provided
        if config_path:
            with open(config_path) as f:
                config_dict = json.load(f)
                # Apply any overrides
                if override_config:
                    config_dict.update(override_config)
                config = OptimizationConfig(**config_dict)
        else:
            # Create minimal config with overrides
            config = OptimizationConfig(**(override_config or {}))

        # Load study
        from pathlib import Path

        study_name = Path(study_path).stem
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")

        pipeline = cls(config)
        return pipeline, study

    def evaluate_on_dataset(
        self, best_params: dict[str, Any], dataset_name: str, dataset_limit: Optional[int] = None
    ) -> dict[str, Any]:
        """Evaluate best parameters on a different dataset.

        Args:
            best_params: Dictionary of hyperparameters to use
            dataset_name: Name of dataset to evaluate on
            dataset_limit: Optional limit on number of samples

        Returns:
            Dictionary containing evaluation results on the new dataset
        """
        # Temporarily override dataset configuration
        original_test_dataset = self.config.test_dataset
        original_test_limit = self.config.test_limit

        self.config.test_dataset = dataset_name
        self.config.test_limit = dataset_limit or self.config.test_limit

        self.logger.info(f"📊 Evaluating on {dataset_name} with {self.config.test_limit} samples")

        # Reload test samples for new dataset
        from . import data_utils

        self.test_samples = data_utils.load_dataset_samples(self.config.test_dataset, self.config.test_limit)

        # Run evaluation
        results = self.evaluate_only(best_params)

        # Restore original config
        self.config.test_dataset = original_test_dataset
        self.config.test_limit = original_test_limit

        return results

    def cleanup_memory(self):
        """Clean up GPU/MPS memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Finish WandB run
        if self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None

        # Clean up device memory
        empty_device_cache(self.device.type)

        import gc

        gc.collect()

    def _init_wandb(self):
        """Initialize WandB for experiment tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict(),
                tags=["optuna", "steering", "optimization"],
                reinit=True,
            )
            self.logger.info(f"WandB initialized: {wandb.run.url}")
        except Exception as e:
            # Don't silently disable - user explicitly requested WandB
            raise OptimizationError(
                reason=f"Failed to initialize WandB: {e}. "
                       f"Run 'wandb login' to authenticate or set use_wandb=False",
                cause=e
            )

    def _log_trial_to_wandb(self, trial: optuna.Trial, metrics: dict[str, float]):
        """Log trial results to WandB."""
        if not self.config.use_wandb or self.wandb_run is None:
            return

        try:
            # Log trial parameters and metrics
            log_data = {f"trial/{k}": v for k, v in trial.params.items()}
            log_data.update({f"metrics/{k}": v for k, v in metrics.items()})
            log_data["trial/number"] = trial.number

            wandb.log(log_data)
        except Exception as e:
            self.logger.warning(f"Failed to log trial to WandB: {e}")

    def _log_final_results_to_wandb(self, study: optuna.Study, final_results: dict[str, Any]):
        """Log final optimization results to WandB."""
        if not self.config.use_wandb or self.wandb_run is None:
            return

        try:
            # Log best trial results
            best_params = {f"best/{k}": v for k, v in study.best_params.items()}
            best_metrics = {
                "best/validation_accuracy": study.best_value,
                "best/baseline_accuracy": final_results["baseline_benchmark_metrics"]["accuracy"],
                "best/steered_accuracy": final_results["steered_benchmark_metrics"]["accuracy"],
                "best/accuracy_improvement": final_results["accuracy_improvement"],
                "study/n_trials": len(study.trials),
                "study/n_complete_trials": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                ),
            }

            wandb.log({**best_params, **best_metrics})

            # Log optimization history
            trial_values = [t.value for t in study.trials if t.value is not None]
            if trial_values:
                wandb.log(
                    {
                        "optimization/best_value_so_far": max(trial_values),
                        "optimization/mean_trial_value": np.mean(trial_values),
                        "optimization/std_trial_value": np.std(trial_values),
                    }
                )

        except Exception as e:
            self.logger.warning(f"Failed to log final results to WandB: {e}")

    def _should_use_multiple_choice_evaluation(self) -> bool:
        """Determine if we should use multiple choice evaluation for this dataset."""
        # Use multiple choice evaluation for TruthfulQA and other MC tasks
        return self.config.test_dataset.lower() in ["truthfulqa_mc1", "truthfulqa", "mmlu"]


def main():
    """Main entry point for optimization pipeline."""
    import argparse
    import json
    
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

    # Run optimization
    pipeline = OptimizationPipeline(config)
    try:
        results = pipeline.run_optimization()

        print("🎉 Optimization completed!")
        print(f"Best validation score: {results['best_validation_score']:.4f}")
        print(f"Test accuracy: {results['steered_benchmark_metrics']['accuracy']:.4f}")
        print(f"Accuracy improvement: {results['accuracy_improvement']:+.4f}")

    finally:
        # Clean up memory
        pipeline.cleanup_memory()


if __name__ == "__main__":
    main()
