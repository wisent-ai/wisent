"""
LoRA DPO training: dataset creation and training pipeline.

Extracted from lora_dpo.py to keep files under 300 lines.
"""
from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOTrainer, DPOConfig

from wisent.core.constants import (
    LORA_DEFAULT_DROPOUT,
    DPO_DEFAULT_BETA,
    COMPARISON_DEFAULT_BATCH_SIZE,
    COMPARISON_NUM_PAIRS, DEFAULT_WEIGHT_DECAY, TRAINING_WARMUP_RATIO,
    COMPARISON_LOGGING_STEPS, LORA_DPO_LEARNING_RATE, LORA_DPO_NUM_EPOCHS,
    JSON_INDENT,
)
from wisent.comparison.utils import (
    generate_contrastive_pairs,
    load_model_and_tokenizer,
)
from wisent.core.utils import preferred_dtype


def create_dpo_dataset(pairs: list[dict], tokenizer) -> Dataset:
    """
    Convert contrastive pairs to DPO dataset format.

    - Chat models: returns conversational format
    - Base models: returns text format with simple formatting
    """
    data = {"prompt": [], "chosen": [], "rejected": []}
    for pair in pairs:
        prompt = pair["prompt"]
        chosen = pair["positive_response"]["model_response"]
        rejected = pair["negative_response"]["model_response"]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            data["prompt"].append([{"role": "user", "content": prompt}])
            data["chosen"].append([{"role": "assistant", "content": chosen}])
            data["rejected"].append([{"role": "assistant", "content": rejected}])
        else:
            data["prompt"].append(f"{prompt}\n")
            data["chosen"].append(chosen)
            data["rejected"].append(rejected)
    return Dataset.from_dict(data)


def train_lora_dpo(
    task: str,
    model_name: str,
    output_path: str | Path,
    lora_r: int,
    lora_alpha: int,
    num_pairs: int = COMPARISON_NUM_PAIRS,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    lora_dropout: float = LORA_DEFAULT_DROPOUT,
    learning_rate: float = LORA_DPO_LEARNING_RATE,
    num_epochs: int = LORA_DPO_NUM_EPOCHS,
    batch_size: int = COMPARISON_DEFAULT_BATCH_SIZE,
    max_length: int | None = None,
    max_prompt_length: int | None = None,
    beta: float = DPO_DEFAULT_BETA,
) -> Path:
    """Train a LoRA adapter using DPO on contrastive pairs."""
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"Step 1: Generating {num_pairs} preference pairs from {task}")
    print(f"{'='*60}")
    pairs, pairs_file = generate_contrastive_pairs(task, num_pairs)
    print(f"Generated {len(pairs)} preference pairs")

    print(f"\n{'='*60}")
    print(f"Step 2: Loading model {model_name}")
    print(f"{'='*60}")
    model, tokenizer = load_model_and_tokenizer(model_name, device, eval_mode=False)
    if max_length is None:
        max_length = tokenizer.model_max_length
    if max_prompt_length is None:
        max_prompt_length = tokenizer.model_max_length // 2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"\n{'='*60}")
    print(f"Step 3: Creating DPO dataset")
    print(f"{'='*60}")
    dataset = create_dpo_dataset(pairs, tokenizer)
    print(f"Dataset size: {len(dataset)}")

    print(f"\n{'='*60}")
    print(f"Step 4: Configuring LoRA (r={lora_r}, alpha={lora_alpha})")
    print(f"{'='*60}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n{'='*60}")
    print(f"Step 5: Configuring DPO training")
    print(f"{'='*60}")
    training_output_dir = tempfile.mkdtemp(prefix="lora_dpo_training_")
    dtype = preferred_dtype(device)
    training_args = DPOConfig(
        output_dir=training_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY, warmup_ratio=TRAINING_WARMUP_RATIO,
        logging_steps=COMPARISON_LOGGING_STEPS, save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        max_length=max_length, max_prompt_length=max_prompt_length,
        beta=beta, loss_type="sigmoid",
    )
    print(f"Beta: {beta}, Max length: {max_length}")
    print(f"Learning rate: {learning_rate}, Epochs: {num_epochs}, Batch: {batch_size}")

    print(f"\n{'='*60}")
    print(f"Step 6: Training with DPO")
    print(f"{'='*60}")
    trainer = DPOTrainer(
        model=model, args=training_args,
        train_dataset=dataset, processing_class=tokenizer,
    )
    trainer.train()

    print(f"\n{'='*60}")
    print(f"Step 7: Saving LoRA adapter")
    print(f"{'='*60}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    metadata = {
        "task": task, "model": model_name, "training_method": "dpo",
        "num_pairs": len(pairs), "lora_r": lora_r, "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout, "learning_rate": learning_rate,
        "num_epochs": num_epochs, "batch_size": batch_size,
        "max_length": max_length, "max_prompt_length": max_prompt_length,
        "beta": beta,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=JSON_INDENT)

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
