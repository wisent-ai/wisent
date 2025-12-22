"""
LoRA fine-tuning method for comparison experiments.

Trains a LoRA adapter on benchmark tasks using supervised fine-tuning (SFT)
on positive responses from contrastive pairs.
"""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel

__all__ = ["train_lora_adapter", "evaluate_lora", "apply_lora_to_model", "remove_lora"]


# Default LoRA configurations per model architecture
LORA_TARGET_MODULES = {
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    "gpt_neo": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "default": "all-linear",
}


def get_target_modules(model_name: str) -> str | list[str]:
    """Get LoRA target modules based on model architecture."""
    model_name_lower = model_name.lower()

    for arch, modules in LORA_TARGET_MODULES.items():
        if arch in model_name_lower:
            return modules

    return LORA_TARGET_MODULES["default"]


def prepare_sft_dataset(
    pairs: list[dict],
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """
    Prepare dataset for SFT from contrastive pairs.

    Uses only positive responses for training.

    Args:
        pairs: List of contrastive pairs
        tokenizer: Tokenizer for formatting
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset ready for SFTTrainer
    """
    formatted_examples = []

    for pair in pairs:
        prompt = pair["prompt"]
        positive_response = pair["positive_response"]["model_response"]

        # Format as chat if tokenizer supports it, otherwise simple format
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": positive_response},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Simple format for base models
            text = f"Q: {prompt}\nA: {positive_response}"

        formatted_examples.append({"text": text})

    return Dataset.from_list(formatted_examples)


def train_lora_adapter(
    task: str,
    model_name: str,
    output_path: str | Path,
    trait_label: str = "correctness",
    num_pairs: int = 50,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    # LoRA-specific parameters
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    max_length: int = 512,
) -> Path:
    """
    Train a LoRA adapter using SFT on positive responses.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        model_name: HuggingFace model name
        output_path: Where to save the LoRA adapter
        trait_label: Label for the trait being trained
        num_pairs: Number of training examples to use
        device: Device to train on
        keep_intermediate: Whether to keep intermediate files
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Maximum sequence length

    Returns:
        Path to the saved LoRA adapter directory
    """
    import gc

    output_path = Path(output_path)

    # Step 1: Generate contrastive pairs
    print(f"Step 1: Generating training data from task: {task}")
    from wisent.core.cli.generate_pairs_from_task import execute_generate_pairs_from_task

    pairs_file = tempfile.NamedTemporaryFile(mode='w', suffix='_pairs.json', delete=False).name
    pairs_args = Namespace(
        task_name=task,
        limit=num_pairs,
        output=pairs_file,
        seed=42,
        verbose=False,
    )
    execute_generate_pairs_from_task(pairs_args)

    with open(pairs_file) as f:
        pairs_data = json.load(f)
    pairs = pairs_data["pairs"]
    print(f"   Loaded {len(pairs)} training examples")

    # Step 2: Load model and tokenizer
    print(f"\nStep 2: Loading model {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Configure LoRA
    print(f"\nStep 3: Configuring LoRA (r={lora_r}, alpha={lora_alpha})...")

    target_modules = get_target_modules(model_name)
    print(f"   Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 4: Prepare dataset
    print(f"\nStep 4: Preparing SFT dataset...")
    train_dataset = prepare_sft_dataset(pairs, tokenizer, max_length=max_length)
    print(f"   Dataset size: {len(train_dataset)} examples")

    # Step 5: Training
    print(f"\nStep 5: Training LoRA adapter...")

    # Create temporary directory for training outputs
    training_output_dir = tempfile.mkdtemp(prefix="lora_training_")

    training_args = SFTConfig(
        output_dir=training_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",  # Don't save checkpoints
        fp16=True,
        report_to="none",  # Disable wandb/tensorboard
        dataset_text_field="text",  # Field containing the text to train on
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Step 6: Save LoRA adapter
    print(f"\nStep 6: Saving LoRA adapter to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save metadata
    metadata = {
        "method": "lora",
        "model": model_name,
        "task": task,
        "trait_label": trait_label,
        "num_pairs": len(pairs),
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules if isinstance(target_modules, list) else [target_modules],
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
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if not keep_intermediate:
        import os
        os.unlink(pairs_file)
        import shutil
        shutil.rmtree(training_output_dir, ignore_errors=True)

    print(f"\nLoRA adapter saved to {output_path}")
    return output_path


def apply_lora_to_model(wisent_model: "WisentModel", lora_path: str | Path) -> None:
    """
    Apply a trained LoRA adapter to a WisentModel.

    Args:
        wisent_model: WisentModel instance
        lora_path: Path to the saved LoRA adapter
    """
    from peft import PeftModel

    lora_path = Path(lora_path)

    # Check if model already has adapters
    if hasattr(wisent_model.hf_model, 'peft_config'):
        # Model already has PEFT, just load new adapter
        wisent_model.hf_model.load_adapter(str(lora_path), adapter_name="steering")
        wisent_model.hf_model.set_adapter("steering")
    else:
        # Wrap model with PEFT
        wisent_model.hf_model = PeftModel.from_pretrained(
            wisent_model.hf_model,
            str(lora_path),
            adapter_name="steering",
        )

    print(f"LoRA adapter loaded from {lora_path}")


def remove_lora(wisent_model: "WisentModel") -> None:
    """
    Remove/disable LoRA adapter from a WisentModel.

    Args:
        wisent_model: WisentModel instance with LoRA applied
    """
    if hasattr(wisent_model.hf_model, 'disable_adapters'):
        try:
            wisent_model.hf_model.disable_adapters()
            print("LoRA adapter disabled")
        except ValueError:
            # No adapter was loaded
            pass
    elif hasattr(wisent_model.hf_model, 'base_model'):
        # Unwrap the model
        wisent_model.hf_model = wisent_model.hf_model.base_model.model
        print("LoRA adapter removed")


def evaluate_lora(
    model_name: str,
    lora_path: str | Path,
    task: str,
    train_ratio: float = 0.8,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_batch_size: int = 8,
    limit: int | None = None,
    output_dir: str | Path = None,
) -> dict:
    """
    Evaluate a trained LoRA adapter comparing base vs LoRA performance.

    Args:
        model_name: HuggingFace model name
        lora_path: Path to trained LoRA adapter
        task: lm-eval task name
        train_ratio: Train/test split ratio
        device: Device to run on
        batch_size: Batch size for evaluation
        max_batch_size: Max batch size
        limit: Limit number of eval examples
        output_dir: Where to save results

    Returns:
        Dict with evaluation results
    """
    import gc
    from lm_eval import evaluator as lm_evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict

    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.utils.dataset_splits import get_test_docs
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
    from wisent.core.evaluators.benchmark_specific.log_likelihoods_evaluator import LogLikelihoodsEvaluator

    lora_path = Path(lora_path)

    # Create test task
    print(f"\n{'='*60}")
    print(f"Creating test task for: {task}")
    print(f"{'='*60}")

    task_dict = get_task_dict([task])
    task_obj = task_dict[task]
    test_docs = get_test_docs(task_obj, benchmark_name=task, train_ratio=train_ratio)
    test_pct = round((1 - train_ratio) * 100)
    print(f"Test split size: {len(test_docs)} docs ({test_pct}% of pooled data)")

    task_obj.test_docs = lambda: test_docs
    task_obj.has_test_docs = lambda: True
    task_obj._eval_docs = test_docs

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)

    # Helper functions
    def run_lm_eval():
        lm = HFLM(
            pretrained=wisent_model.hf_model,
            tokenizer=wisent_model.tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
        results = lm_evaluator.evaluate(lm=lm, task_dict=task_dict, limit=limit)
        task_results = results.get("results", {}).get(task, {})
        for key in ["acc", "acc,none", "accuracy", "acc_norm", "acc_norm,none"]:
            if key in task_results:
                return task_results[key]
        return 0.0

    def run_ll_eval():
        ll_evaluator = LogLikelihoodsEvaluator()
        extractor = get_extractor(task)
        docs = list(task_obj.test_docs())
        if limit:
            docs = docs[:limit]

        correct = 0
        for i, doc in enumerate(docs):
            question = task_obj.doc_to_text(doc)
            choices, expected = extractor.extract_choices_and_answer(task_obj, doc)
            result = ll_evaluator.evaluate(
                response="",
                expected=expected,
                model=wisent_model,
                question=question,
                choices=choices,
                task_name=task,
            )
            if result.ground_truth == "TRUTHFUL":
                correct += 1
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(docs)}, acc: {correct/(i+1):.4f}")

        return correct / len(docs) if docs else 0.0

    # BASE evaluation
    print(f"\n{'='*60}")
    print(f"Running BASE evaluation (no LoRA)")
    print(f"{'='*60}")

    base_acc_lm_eval = run_lm_eval()
    print(f"Base accuracy (lm-eval): {base_acc_lm_eval:.4f}")

    base_acc_ll = run_ll_eval()
    print(f"Base accuracy (LL): {base_acc_ll:.4f}")

    # Apply LoRA
    print(f"\n{'='*60}")
    print(f"Applying LoRA adapter from: {lora_path}")
    print(f"{'='*60}")
    apply_lora_to_model(wisent_model, lora_path)

    # LORA evaluation
    print(f"\n{'='*60}")
    print(f"Running LORA evaluation")
    print(f"{'='*60}")

    lora_acc_lm_eval = run_lm_eval()
    print(f"LoRA accuracy (lm-eval): {lora_acc_lm_eval:.4f}")

    lora_acc_ll = run_ll_eval()
    print(f"LoRA accuracy (LL): {lora_acc_ll:.4f}")

    # Cleanup
    remove_lora(wisent_model)
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Results
    results = {
        "task": task,
        "model": model_name,
        "lora_path": str(lora_path),
        "train_ratio": train_ratio,
        "base_accuracy_lm_eval": base_acc_lm_eval,
        "base_accuracy_ll": base_acc_ll,
        "lora_accuracy_lm_eval": lora_acc_lm_eval,
        "lora_accuracy_ll": lora_acc_ll,
        "difference_lm_eval": lora_acc_lm_eval - base_acc_lm_eval,
        "difference_ll": lora_acc_ll - base_acc_ll,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"LoRA: {lora_path}")
    print(f"{'-'*60}")
    print(f"{'Metric':<20} {'Base':<12} {'LoRA':<12} {'Diff':<12}")
    print(f"{'-'*60}")
    print(f"{'lm-eval accuracy':<20} {base_acc_lm_eval:<12.4f} {lora_acc_lm_eval:<12.4f} {lora_acc_lm_eval - base_acc_lm_eval:+.4f}")
    print(f"{'LL accuracy':<20} {base_acc_ll:<12.4f} {lora_acc_ll:<12.4f} {lora_acc_ll - base_acc_ll:+.4f}")
    print(f"{'='*60}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_lora_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate LoRA adapter on benchmark task")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of training examples")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    # Eval args
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")

    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.task}_lora_adapter"

    # Train
    train_lora_adapter(
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
    )

    # Evaluate
    if not args.skip_eval:
        evaluate_lora(
            model_name=args.model,
            lora_path=output_path,
            task=args.task,
            train_ratio=args.train_ratio,
            device=args.device,
            batch_size=args.eval_batch_size,
            limit=args.eval_limit,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
