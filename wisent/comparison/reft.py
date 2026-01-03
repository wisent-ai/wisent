"""
ReFT (Representation Fine-Tuning) method for comparison experiments.

Trains a LoReFT intervention on benchmark tasks using supervised fine-tuning (SFT)
on positive responses from contrastive pairs.

LoReFT operates on hidden representations rather than weights, making it
10-50x more parameter-efficient than LoRA.

Based on: "ReFT: Representation Finetuning for Language Models" (arXiv:2404.03592)
Uses pyreft library from Stanford NLP.
"""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset

from wisent.comparison.utils import (
    generate_contrastive_pairs,
    create_test_only_task,
    run_ll_evaluation,
    load_model_and_tokenizer,
    apply_steering_to_model,
    remove_steering,
)
from wisent.core.utils.device import preferred_dtype

import pyreft
import transformers

from wisent.core.models.wisent_model import WisentModel

__all__ = ["train_reft_adapter", "evaluate_reft", "apply_reft_to_model", "remove_reft"]


def prepare_reft_dataset(
    pairs: list[dict],
    tokenizer,
    max_length: int = 512,
) -> tuple[list[str], list[str]]:
    """
    Prepare dataset for ReFT training from contrastive pairs.

    Uses only positive responses for training.

    Args:
        pairs: List of contrastive pairs
        tokenizer: Tokenizer for formatting
        max_length: Maximum sequence length

    Returns:
        Tuple of (prompts, responses) lists
    """
    prompts = []
    responses = []

    for pair in pairs:
        prompt = pair["prompt"]
        positive_response = pair["positive_response"]["model_response"]

        # Format as chat if tokenizer supports it
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            # For chat models, format as conversation
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Simple format for base models
            formatted_prompt = f"{prompt}\n"

        prompts.append(formatted_prompt)
        responses.append(positive_response)

    return prompts, responses


def train_reft_adapter(
    task: str,
    model_name: str,
    output_path: str | Path,
    trait_label: str = "correctness",
    num_pairs: int = 50,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    # ReFT-specific parameters
    low_rank_dimension: int = 4,
    intervention_layers: str | None = None,
    learning_rate: float = 5e-4,
    num_epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 512,
) -> Path:
    """
    Train a LoReFT intervention using SFT on positive responses.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        model_name: HuggingFace model name
        output_path: Where to save the ReFT intervention
        trait_label: Label for the trait being trained
        num_pairs: Number of training examples to use
        device: Device to train on
        keep_intermediate: Whether to keep intermediate files
        low_rank_dimension: Rank for LoReFT (default: 4, very small!)
        intervention_layers: Comma-separated layers or None for default
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Maximum sequence length

    Returns:
        Path to the saved ReFT intervention directory
    """
    import transformers
    import pyreft

    # Monkey-patch pyreft to work with newer transformers (Issue #165)
    # transformers>=4.46 passes num_items_in_batch to compute_loss()
    _original_compute_loss = pyreft.ReftTrainer.compute_loss

    def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return _original_compute_loss(self, model, inputs, return_outputs)

    pyreft.ReftTrainer.compute_loss = _patched_compute_loss

    output_path = Path(output_path)

    # Step 1: Generate contrastive pairs
    print(f"Step 1: Generating training data from task: {task}")
    pairs, pairs_file = generate_contrastive_pairs(task, num_pairs)
    print(f"   Loaded {len(pairs)} training examples")

    # Step 2: Load model and tokenizer
    print(f"\nStep 2: Loading model {model_name}...")
    dtype = preferred_dtype(device)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Parse intervention layers
    if intervention_layers is None:
        layer_indices = [get_default_layer(model_name)]
    else:
        layer_indices = [int(l.strip()) for l in intervention_layers.split(",")]

    print(f"\nStep 3: Configuring LoReFT (rank={low_rank_dimension}, layers={layer_indices})...")

    # Step 4: Create ReFT config and model
    # Get hidden size from model config
    hidden_size = model.config.hidden_size

    # Create interventions for each layer
    representations = []
    for layer_idx in layer_indices:
        representations.append({
            "layer": layer_idx,
            "component": "block_output",
            "low_rank_dimension": low_rank_dimension,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=hidden_size,
                low_rank_dimension=low_rank_dimension,
                dtype=dtype,
            ),
        })

    reft_config = pyreft.ReftConfig(representations=representations)
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)
    reft_model.print_trainable_parameters()

    # Step 5: Prepare dataset
    print(f"\nStep 5: Preparing ReFT dataset...")
    prompts, responses = prepare_reft_dataset(pairs, tokenizer, max_length=max_length)
    print(f"   Dataset size: {len(prompts)} examples")

    # Create data module for ReFT training
    # ReFT expects data in specific format with intervention positions
    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer, model, prompts, responses
    )

    # Step 6: Training
    print(f"\nStep 6: Training LoReFT intervention...")

    training_output_dir = tempfile.mkdtemp(prefix="reft_training_")

    training_args = transformers.TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()

    # Step 7: Save ReFT intervention
    print(f"\nStep 7: Saving ReFT intervention to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    reft_model.save(output_path)
    tokenizer.save_pretrained(output_path)

    # Save metadata
    metadata = {
        "method": "reft",
        "model": model_name,
        "task": task,
        "trait_label": trait_label,
        "num_pairs": len(pairs),
        "reft_config": {
            "low_rank_dimension": low_rank_dimension,
            "intervention_layers": layer_indices,
            "component": "block_output",
        },
        "training_config": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "max_length": max_length,
        },
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Cleanup
    del reft_model, trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if not keep_intermediate:
        import os
        os.unlink(pairs_file)
        import shutil
        shutil.rmtree(training_output_dir, ignore_errors=True)

    print(f"\nReFT intervention saved to {output_path}")
    return output_path


def apply_reft_to_model(wisent_model: "WisentModel", reft_path: str | Path) -> None:
    """
    Apply a trained ReFT intervention to a WisentModel.

    Args:
        wisent_model: WisentModel instance
        reft_path: Path to the saved ReFT intervention
    """
    import pyreft

    reft_path = Path(reft_path)

    # Load ReFT model wrapping the existing model
    reft_model = pyreft.ReftModel.load(
        str(reft_path),
        wisent_model.hf_model,
    )
    reft_model.set_device(wisent_model.device)

    # Ensure intervention dtype matches model dtype
    model_dtype = next(wisent_model.hf_model.parameters()).dtype
    for k, v in reft_model.interventions.items():
        intervention = v[0] if isinstance(v, (list, tuple)) else v
        if hasattr(intervention, 'to'):
            intervention.to(dtype=model_dtype)

    # Create wrapper to translate HF call signature to pyvene's signature
    # pyvene expects: model(base={"input_ids": ...})
    # HF models expect: model(input_ids=...)
    class ReftModelWrapper(torch.nn.Module):
        def __init__(self, reft_model, original_model):
            super().__init__()
            self._reft_model = reft_model
            self._original_model = original_model
            # Copy essential attributes from original model
            self.config = original_model.config
            self.device = next(original_model.parameters()).device

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # Build base dict for pyvene
            base = {}
            if input_ids is not None:
                base["input_ids"] = input_ids
            if attention_mask is not None:
                base["attention_mask"] = attention_mask
            # Add other kwargs to base
            base.update(kwargs)
            # Call ReftModel - returns (base_output, intervened_output)
            _, outputs = self._reft_model(base=base)
            return outputs

        def __getattr__(self, name):
            if name in ("_reft_model", "_original_model", "config", "device",
                        "training", "_parameters", "_buffers", "_modules"):
                return super().__getattr__(name)
            return getattr(self._original_model, name)

    wrapper = ReftModelWrapper(reft_model, wisent_model.hf_model)

    # Store original model and replace with wrapper
    wisent_model._original_model = wisent_model.hf_model
    wisent_model.hf_model = wrapper

    print(f"ReFT intervention loaded from {reft_path}")


def remove_reft(wisent_model: "WisentModel") -> None:
    """
    Remove/disable ReFT intervention from a WisentModel.

    Args:
        wisent_model: WisentModel instance with ReFT applied
    """
    if hasattr(wisent_model, '_original_model'):
        wisent_model.hf_model = wisent_model._original_model
        del wisent_model._original_model
        print("ReFT intervention removed")
    else:
        print("No ReFT intervention to remove")


def evaluate_reft(
    model_name: str,
    reft_path: str | Path,
    task: str,
    train_ratio: float = 0.8,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_batch_size: int = 8,
    limit: int | None = None,
    output_dir: str | Path = None,
    # Training metadata (for output)
    num_train_pairs: int | None = None,
    num_epochs: int | None = None,
    low_rank_dimension: int | None = None,
    intervention_layers: list[int] | None = None,
    learning_rate: float | None = None,
    # Steering parameters (optional)
    with_steering: bool = False,
    steering_method: str = "caa",
    steering_layers: str = "12",
    steering_num_pairs: int = 50,
    steering_scales: list[float] | None = None,
    extraction_strategy: str = "mc_completion",
) -> dict:
    """
    Evaluate a trained ReFT intervention comparing base vs ReFT performance.

    Optionally also evaluates ReFT + steering at multiple scales.

    Args:
        model_name: HuggingFace model name
        reft_path: Path to trained ReFT intervention
        task: lm-eval task name
        train_ratio: Train/test split ratio
        device: Device to run on
        batch_size: Batch size for evaluation
        max_batch_size: Max batch size
        limit: Limit number of eval examples
        output_dir: Where to save results
        with_steering: Whether to also evaluate ReFT + steering
        steering_method: Steering method (caa or fgaa)
        steering_layers: Layers for steering vector
        steering_num_pairs: Number of pairs for steering generation
        steering_scales: List of steering scales to evaluate
        extraction_strategy: Strategy for activation extraction

    Returns:
        Dict with evaluation results
    """


    reft_path = Path(reft_path)

    if steering_scales is None:
        steering_scales = [1.0, 2.0, 4.0]

    # Create test task
    print(f"\n{'='*60}")
    print(f"Creating test task for: {task}")
    print(f"{'='*60}")

    task_dict = create_test_only_task(task, train_ratio=train_ratio)

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)

    # BASE evaluation
    print(f"\n{'='*60}")
    print(f"Running BASE evaluation (no ReFT)")
    print(f"{'='*60}")

    base_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"Base accuracy: {base_acc_ll:.4f}")

    # Apply ReFT
    print(f"\n{'='*60}")
    print(f"Applying ReFT intervention from: {reft_path}")
    print(f"{'='*60}")
    apply_reft_to_model(wisent_model, reft_path)

    # REFT evaluation
    print(f"\n{'='*60}")
    print(f"Running REFT evaluation")
    print(f"{'='*60}")

    reft_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"ReFT accuracy: {reft_acc_ll:.4f}")

    # Results dict
    results = {
        "task": task,
        "model": model_name,
        "reft_path": str(reft_path),
        # Training config
        "num_train_pairs": num_train_pairs,
        "num_epochs": num_epochs,
        "low_rank_dimension": low_rank_dimension,
        "intervention_layers": intervention_layers,
        "learning_rate": learning_rate,
        # Eval config
        "train_ratio": train_ratio,
        "eval_limit": limit,
        # Results
        "base_accuracy": base_acc_ll,
        "reft_accuracy": reft_acc_ll,
        "reft_diff": reft_acc_ll - base_acc_ll,
    }

    # ReFT + Steering evaluation (if enabled)
    if with_steering:
        from wisent.core.trainers.steering_trainer import WisentSteeringTrainer
        from wisent.core.steering_methods import get_steering_method
        from wisent.core.activations.extraction_strategy import ExtractionStrategy
        from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
        from wisent.core.contrastive_pairs.core.pair import ContrastivePair
        from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse

        # Generate contrastive pairs for steering
        print(f"\n{'='*60}")
        print(f"Generating {steering_num_pairs} contrastive pairs for steering")
        print(f"{'='*60}")
        pairs_data, pairs_file = generate_contrastive_pairs(task, steering_num_pairs)

        # Convert to ContrastivePairSet
        pairs = []
        for p in pairs_data:
            pair = ContrastivePair(
                prompt=p["prompt"],
                positive_response=PositiveResponse(model_response=p["positive_response"]["model_response"]),
                negative_response=NegativeResponse(model_response=p["negative_response"]["model_response"]),
            )
            pairs.append(pair)
        pair_set = ContrastivePairSet(pairs=pairs, name=f"{task}_reft_steering")
        print(f"Created {len(pair_set)} contrastive pairs")

        # Generate steering vector on ReFT model
        print(f"\n{'='*60}")
        print(f"Generating {steering_method.upper()} steering vector on ReFT model")
        print(f"Layers: {steering_layers}")
        print(f"{'='*60}")

        steering_method_obj = get_steering_method(steering_method, device=device)
        strategy = ExtractionStrategy(extraction_strategy)

        trainer = WisentSteeringTrainer(
            model=wisent_model,
            pair_set=pair_set,
            steering_method=steering_method_obj,
        )

        result = trainer.run(
            layers_spec=steering_layers,
            strategy=strategy,
            accept_low_quality_vector=True,
        )

        # Convert to dict format for apply_steering_to_model
        steering_vectors = {}
        for layer_name, tensor in result.steered_vectors.to_dict().items():
            if tensor is not None:
                steering_vectors[layer_name] = tensor.cpu().float().tolist()

        steering_data = {
            "steering_vectors": steering_vectors,
            "layers": list(steering_vectors.keys()),
        }

        # Cleanup temp file
        import os
        os.unlink(pairs_file)

        # Add steering info to results
        results["steering"] = {
            "method": steering_method,
            "layers": list(steering_vectors.keys()),
            "num_pairs": steering_num_pairs,
            "extraction_strategy": extraction_strategy,
            "scales": {},
        }

        # Evaluate at each scale
        for scale in steering_scales:
            print(f"\n{'='*60}")
            print(f"Evaluating ReFT+{steering_method.upper()} at scale={scale}")
            print(f"{'='*60}")

            apply_steering_to_model(wisent_model, steering_data, scale=scale)

            steer_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
            print(f"ReFT+{steering_method.upper()} accuracy: {steer_acc_ll:.4f}")

            remove_steering(wisent_model)

            results["steering"]["scales"][str(scale)] = {
                "accuracy": steer_acc_ll,
                "diff_from_base": steer_acc_ll - base_acc_ll,
                "diff_from_reft": steer_acc_ll - reft_acc_ll,
            }

    # Cleanup
    remove_reft(wisent_model)
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"ReFT: {reft_path}")
    print(f"{'-'*50}")
    print(f"{'Method':<30} {'Accuracy':<12} {'Diff':<10}")
    print(f"{'-'*50}")
    print(f"{'Base':<30} {base_acc_ll:<12.4f} {'':<10}")
    print(f"{'ReFT':<30} {reft_acc_ll:<12.4f} {reft_acc_ll - base_acc_ll:+.4f}")

    if with_steering:
        for scale, res in results["steering"]["scales"].items():
            label = f"ReFT+{steering_method.upper()}@{scale}"
            print(f"{label:<30} {res['accuracy']:<12.4f} {res['diff_from_base']:+.4f}")

    print(f"{'='*50}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        model_dir_name = model_name.replace("/", "_")
        output_dir = output_dir / model_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_reft_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate ReFT intervention on benchmark task")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of training examples")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--low-rank-dimension", type=int, default=4, help="LoReFT rank (default: 4)")
    parser.add_argument("--intervention-layers", default=None, help="Comma-separated intervention layers (default: auto)")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    # Eval args
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", default="auto", help="Eval batch size (int or 'auto')")
    parser.add_argument("--eval-max-batch-size", type=int, default=64, help="Max eval batch size for auto")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    # ReFT + Steering args
    parser.add_argument("--with-steering", action="store_true", help="Also evaluate ReFT + steering")
    parser.add_argument("--steering-method", default="caa", choices=["caa", "fgaa"], help="Steering method")
    parser.add_argument("--steering-layers", default="12", help="Layers for steering vector")
    parser.add_argument("--steering-num-pairs", type=int, default=50, help="Number of pairs for steering")
    parser.add_argument("--steering-scales", default="1.0,2.0,4.0", help="Comma-separated steering scales")
    parser.add_argument("--extraction-strategy", default="mc_completion", help="Extraction strategy for steering")

    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.task}_reft_intervention"

    # Parse intervention layers for metadata
    if args.intervention_layers:
        intervention_layers = [int(l.strip()) for l in args.intervention_layers.split(",")]
    else:
        intervention_layers = [get_default_layer(args.model)]

    # Train
    train_reft_adapter(
        task=args.task,
        model_name=args.model,
        output_path=output_path,
        num_pairs=args.num_pairs,
        device=args.device,
        keep_intermediate=args.keep_intermediate,
        low_rank_dimension=args.low_rank_dimension,
        intervention_layers=args.intervention_layers,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Evaluate base vs ReFT (and optionally ReFT + steering)
    if not args.skip_eval:
        # Parse eval batch size (can be "auto" or int)
        eval_batch_size = args.eval_batch_size
        if eval_batch_size != "auto":
            eval_batch_size = int(eval_batch_size)

        # Parse steering scales
        steering_scales = [float(s.strip()) for s in args.steering_scales.split(",")]

        evaluate_reft(
            model_name=args.model,
            reft_path=output_path,
            task=args.task,
            train_ratio=args.train_ratio,
            device=args.device,
            batch_size=eval_batch_size,
            max_batch_size=args.eval_max_batch_size,
            limit=args.eval_limit,
            output_dir=args.output_dir,
            # Training metadata
            num_train_pairs=args.num_pairs,
            num_epochs=args.num_epochs,
            low_rank_dimension=args.low_rank_dimension,
            intervention_layers=intervention_layers,
            learning_rate=args.learning_rate,
            # Steering parameters
            with_steering=args.with_steering,
            steering_method=args.steering_method,
            steering_layers=args.steering_layers,
            steering_num_pairs=args.steering_num_pairs,
            steering_scales=steering_scales,
            extraction_strategy=args.extraction_strategy,
        )


if __name__ == "__main__":
    main()
