"""
LoRA fine-tuning using DPO (Direct Preference Optimization).

Unlike SFT which trains on positive examples only, DPO trains on
preference pairs (chosen vs rejected) to directly optimize for preferences.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOTrainer, DPOConfig

from wisent.comparison.utils import (
    generate_contrastive_pairs,
    create_test_only_task,
    extract_accuracy,
    run_lm_eval_evaluation,
    run_ll_evaluation,
    load_model_and_tokenizer,
    apply_steering_to_model,
    remove_steering,
)
from wisent.core.utils.device import preferred_dtype

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel


def create_dpo_dataset(pairs: list[dict]) -> Dataset:
    """
    Convert contrastive pairs to DPO dataset format.

    DPO expects:
    - prompt: the input prompt
    - chosen: the preferred response
    - rejected: the non-preferred response
    """
    data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for pair in pairs:
        prompt = pair["prompt"]
        chosen = pair["positive_response"]["model_response"]
        rejected = pair["negative_response"]["model_response"]

        data["prompt"].append(prompt)
        data["chosen"].append(chosen)
        data["rejected"].append(rejected)

    return Dataset.from_dict(data)


def train_lora_dpo(
    task: str,
    model_name: str,
    output_path: str | Path,
    num_pairs: int = 50,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 512,
    max_prompt_length: int = 256,
    beta: float = 0.1,
) -> Path:
    """
    Train a LoRA adapter using DPO on contrastive pairs from an lm-eval task.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        model_name: HuggingFace model name
        output_path: Where to save the trained adapter
        num_pairs: Number of preference pairs to use
        device: Device to run on
        keep_intermediate: Whether to keep intermediate files
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Max total sequence length
        max_prompt_length: Max prompt length
        beta: DPO beta parameter (controls deviation from reference model)

    Returns:
        Path to saved adapter
    """
    output_path = Path(output_path)

    # Step 1: Generate contrastive pairs
    print(f"\n{'='*60}")
    print(f"Step 1: Generating {num_pairs} preference pairs from {task}")
    print(f"{'='*60}")

    pairs, pairs_file = generate_contrastive_pairs(task, num_pairs)
    print(f"Generated {len(pairs)} preference pairs")

    # Step 2: Create DPO dataset
    print(f"\n{'='*60}")
    print(f"Step 2: Creating DPO dataset")
    print(f"{'='*60}")

    dataset = create_dpo_dataset(pairs)
    print(f"Dataset size: {len(dataset)}")

    # Step 3: Load model
    print(f"\n{'='*60}")
    print(f"Step 3: Loading model {model_name}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_name, device, eval_mode=False)

    # Ensure tokenizer has padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO typically uses left padding

    # Step 4: Configure LoRA
    print(f"\n{'='*60}")
    print(f"Step 4: Configuring LoRA (r={lora_r}, alpha={lora_alpha})")
    print(f"{'='*60}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 5: Configure DPO training
    print(f"\n{'='*60}")
    print(f"Step 5: Configuring DPO training")
    print(f"{'='*60}")

    training_output_dir = tempfile.mkdtemp(prefix="lora_dpo_training_")

    # Determine dtype
    dtype = preferred_dtype(device)

    training_args = DPOConfig(
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
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        beta=beta,
        loss_type="sigmoid",  # Standard DPO loss
    )

    print(f"Beta: {beta}")
    print(f"Max length: {max_length}")
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    # Step 6: Train with DPO
    print(f"\n{'='*60}")
    print(f"Step 6: Training with DPO")
    print(f"{'='*60}")

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Step 7: Save adapter
    print(f"\n{'='*60}")
    print(f"Step 7: Saving LoRA adapter")
    print(f"{'='*60}")

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save metadata
    metadata = {
        "task": task,
        "model": model_name,
        "training_method": "dpo",
        "num_pairs": len(pairs),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "beta": beta,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Cleanup
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not keep_intermediate:
        import os
        import shutil
        os.unlink(pairs_file)
        shutil.rmtree(training_output_dir, ignore_errors=True)

    print(f"\nDPO LoRA adapter saved to {output_path}")
    return output_path


def evaluate_lora_dpo(
    model_name: str,
    lora_path: str | Path,
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
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    learning_rate: float | None = None,
    beta: float | None = None,
    max_length: int | None = None,
    max_prompt_length: int | None = None,
    # Steering parameters (optional)
    with_steering: bool = False,
    steering_method: str = "caa",
    steering_layers: str = "12",
    steering_num_pairs: int = 50,
    steering_scales: list[float] | None = None,
    extraction_strategy: str = "mc_completion",
) -> dict:
    """
    Evaluate a trained DPO LoRA adapter.

    Compares base model vs DPO-LoRA model accuracy.
    Optionally also evaluates DPO-LoRA + steering at multiple scales.
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.comparison.lora import apply_lora_to_model, remove_lora

    lora_path = Path(lora_path)

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

    # Base evaluation
    print(f"\n{'='*60}")
    print(f"Running BASE evaluation")
    print(f"{'='*60}")

    base_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, limit)
    base_acc_lm_eval = extract_accuracy(base_results, task)
    print(f"Base accuracy (lm-eval): {base_acc_lm_eval:.4f}")

    base_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"Base accuracy (LL): {base_acc_ll:.4f}")

    # Apply DPO LoRA
    print(f"\n{'='*60}")
    print(f"Applying DPO LoRA adapter from: {lora_path}")
    print(f"{'='*60}")
    apply_lora_to_model(wisent_model, lora_path)

    # LoRA evaluation
    print(f"\n{'='*60}")
    print(f"Running DPO-LORA evaluation")
    print(f"{'='*60}")

    lora_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, limit)
    lora_acc_lm_eval = extract_accuracy(lora_results, task)
    print(f"DPO-LoRA accuracy (lm-eval): {lora_acc_lm_eval:.4f}")

    lora_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"DPO-LoRA accuracy (LL): {lora_acc_ll:.4f}")

    # Results dict
    results = {
        "task": task,
        "model": model_name,
        "training_method": "dpo",
        "lora_path": str(lora_path),
        # Training config
        "num_train_pairs": num_train_pairs,
        "num_epochs": num_epochs,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "beta": beta,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        # Eval config
        "train_ratio": train_ratio,
        "eval_limit": limit,
        # Results
        "base_accuracy_lm_eval": base_acc_lm_eval,
        "base_accuracy_ll": base_acc_ll,
        "lora_accuracy_lm_eval": lora_acc_lm_eval,
        "lora_accuracy_ll": lora_acc_ll,
        "lora_diff_lm_eval": lora_acc_lm_eval - base_acc_lm_eval,
        "lora_diff_ll": lora_acc_ll - base_acc_ll,
    }

    # DPO-LoRA + Steering evaluation (if enabled)
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
        pair_set = ContrastivePairSet(pairs=pairs, name=f"{task}_dpo_lora_steering")
        print(f"Created {len(pair_set)} contrastive pairs")

        # Generate steering vector on DPO-LoRA model
        print(f"\n{'='*60}")
        print(f"Generating {steering_method.upper()} steering vector on DPO-LoRA model")
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
            print(f"Evaluating DPO-LoRA+{steering_method.upper()} at scale={scale}")
            print(f"{'='*60}")

            apply_steering_to_model(wisent_model, steering_data, scale=scale)

            steer_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, limit)
            steer_acc_lm_eval = extract_accuracy(steer_results, task)
            print(f"DPO-LoRA+{steering_method.upper()} accuracy (lm-eval): {steer_acc_lm_eval:.4f}")

            steer_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
            print(f"DPO-LoRA+{steering_method.upper()} accuracy (LL): {steer_acc_ll:.4f}")

            remove_steering(wisent_model)

            results["steering"]["scales"][str(scale)] = {
                "accuracy_lm_eval": steer_acc_lm_eval,
                "accuracy_ll": steer_acc_ll,
                "diff_from_base_lm_eval": steer_acc_lm_eval - base_acc_lm_eval,
                "diff_from_base_ll": steer_acc_ll - base_acc_ll,
                "diff_from_lora_lm_eval": steer_acc_lm_eval - lora_acc_lm_eval,
                "diff_from_lora_ll": steer_acc_ll - lora_acc_ll,
            }

    # Cleanup
    remove_lora(wisent_model)
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"Training: DPO")
    print(f"{'-'*70}")
    print(f"{'Method':<25} {'lm-eval acc':<15} {'LL acc':<15} {'Diff (lm-eval)':<15}")
    print(f"{'-'*70}")
    print(f"{'Base':<25} {base_acc_lm_eval:<15.4f} {base_acc_ll:<15.4f} {'':<15}")
    print(f"{'DPO-LoRA':<25} {lora_acc_lm_eval:<15.4f} {lora_acc_ll:<15.4f} {lora_acc_lm_eval - base_acc_lm_eval:+.4f}")

    if with_steering:
        for scale, res in results["steering"]["scales"].items():
            label = f"DPO-LoRA+{steering_method.upper()}@{scale}"
            print(f"{label:<25} {res['accuracy_lm_eval']:<15.4f} {res['accuracy_ll']:<15.4f} {res['diff_from_base_lm_eval']:+.4f}")

    print(f"{'='*70}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        model_dir_name = model_name.replace("/", "_")
        output_dir = output_dir / model_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_lora_dpo_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LoRA adapter using DPO")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of preference pairs")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max total sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=256, help="Max prompt length")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (controls KL penalty)")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    # Eval args
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", default="auto", help="Eval batch size")
    parser.add_argument("--eval-max-batch-size", type=int, default=64, help="Max eval batch size")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    # DPO-LoRA + Steering args
    parser.add_argument("--with-steering", action="store_true", help="Also evaluate DPO-LoRA + steering")
    parser.add_argument("--steering-method", default="caa", choices=["caa", "fgaa"], help="Steering method")
    parser.add_argument("--steering-layers", default="12", help="Layers for steering vector")
    parser.add_argument("--steering-num-pairs", type=int, default=50, help="Number of pairs for steering")
    parser.add_argument("--steering-scales", default="1.0,2.0,4.0", help="Comma-separated steering scales")
    parser.add_argument("--extraction-strategy", default="mc_balanced", help="Extraction strategy for steering")

    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.task}_lora_dpo_adapter"

    # Train
    train_lora_dpo(
        task=args.task,
        model_name=args.model,
        output_path=output_path,
        num_pairs=args.num_pairs,
        device=args.device,
        keep_intermediate=args.keep_intermediate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
    )

    # Evaluate
    if not args.skip_eval:
        eval_batch_size = args.eval_batch_size
        if eval_batch_size != "auto":
            eval_batch_size = int(eval_batch_size)

        # Parse steering scales
        steering_scales = [float(s.strip()) for s in args.steering_scales.split(",")]

        evaluate_lora_dpo(
            model_name=args.model,
            lora_path=output_path,
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
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
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
