"""LoRA training: prepare_sft_dataset, train_lora_adapter."""
from __future__ import annotations
import gc
import json
import tempfile
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
from wisent.comparison.utils import generate_contrastive_pairs, load_model_and_tokenizer
from wisent.core.utils import preferred_dtype

__all__ = ["prepare_sft_dataset", "get_target_modules", "train_lora_adapter"]

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

def prepare_sft_dataset(pairs: list[dict], tokenizer) -> Dataset:
    """Prepare dataset for SFT from contrastive pairs."""
    formatted_examples = []
    for pair in pairs:
        prompt = pair["prompt"]
        response = pair["positive_response"]["model_response"]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            formatted_examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            })
        else:
            formatted_examples.append({"text": f"{prompt}\n{response}"})
    return Dataset.from_list(formatted_examples)

def train_lora_adapter(
    task: str, model_name: str, output_path: str | Path,
    trait_label: str = "correctness", num_pairs: int = 50,
    device: str = "cuda:0", keep_intermediate: bool = False,
    lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
    learning_rate: float = 2e-4, num_epochs: int = 3,
    batch_size: int = 2, max_length: int = 512,
) -> Path:
    """Train a LoRA adapter using SFT on positive responses."""
    output_path = Path(output_path)
    print(f"Step 1: Generating training data from task: {task}")
    pairs, pairs_file = generate_contrastive_pairs(task, num_pairs)
    print(f"   Loaded {len(pairs)} training examples")
    print(f"\nStep 2: Loading model {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device, eval_mode=False)
    print(f"\nStep 3: Configuring LoRA (r={lora_r}, alpha={lora_alpha})...")
    target_modules = get_target_modules(model_name)
    print(f"   Target modules: {target_modules}")
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                             lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"\nStep 4: Preparing SFT dataset...")
    train_dataset = prepare_sft_dataset(pairs, tokenizer)
    print(f"   Dataset size: {len(train_dataset)} examples")
    print(f"\nStep 5: Training LoRA adapter...")
    training_output_dir = tempfile.mkdtemp(prefix="lora_training_")
    dtype = preferred_dtype(device)
    training_args = SFTConfig(
        output_dir=training_output_dir, num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=1,
        learning_rate=learning_rate, weight_decay=0.01, warmup_ratio=0.1,
        logging_steps=10, save_strategy="no",
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        report_to="none", max_seq_length=max_length,
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset,
                         processing_class=tokenizer)
    trainer.train()
    print(f"\nStep 6: Saving LoRA adapter to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    metadata = {
        "method": "lora", "model": model_name, "task": task, "trait_label": trait_label,
        "num_pairs": len(pairs),
        "lora_config": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
                        "target_modules": target_modules if isinstance(target_modules, list) else [target_modules]},
        "training_config": {"learning_rate": learning_rate, "num_epochs": num_epochs,
                            "batch_size": batch_size, "max_length": max_length},
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if not keep_intermediate:
        import os, shutil
        os.unlink(pairs_file)
        shutil.rmtree(training_output_dir, ignore_errors=True)
    print(f"\nLoRA adapter saved to {output_path}")
    return output_path
