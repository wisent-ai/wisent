"""ReFT training: prepare_reft_dataset and train_reft_adapter."""
from __future__ import annotations
import gc
import json
import tempfile
from pathlib import Path
import torch
from datasets import Dataset
from wisent.comparison.utils import generate_contrastive_pairs
from wisent.core.utils import preferred_dtype
from wisent.core.constants import (
    COMPARISON_NUM_PAIRS, TRAINING_WEIGHT_DECAY, TRAINING_WARMUP_RATIO,
    COMPARISON_REFT_LEARNING_RATE, COMPARISON_NUM_EPOCHS_DEFAULT,
    COMPARISON_TRAINING_BATCH_SIZE, COMPARISON_MAX_LENGTH,
    COMPARISON_LOGGING_STEPS, LOREFT_DEFAULT_RANK,
    GRADIENT_ACCUMULATION_STEPS_DEFAULT,
)
import pyreft
import transformers

__all__ = ["prepare_reft_dataset", "train_reft_adapter"]


def prepare_reft_dataset(
    pairs: list[dict],
    tokenizer,
    max_length: int = COMPARISON_MAX_LENGTH,
) -> tuple[list[str], list[str]]:
    """
    Prepare dataset for ReFT training from contrastive pairs.
    Uses only positive responses for training.
    """
    prompts = []
    responses = []
    for pair in pairs:
        prompt = pair["prompt"]
        positive_response = pair["positive_response"]["model_response"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            formatted_prompt = f"{prompt}\n"
        prompts.append(formatted_prompt)
        responses.append(positive_response)
    return prompts, responses


def train_reft_adapter(
    task: str,
    model_name: str,
    output_path: str | Path,
    trait_label: str = "correctness",
    num_pairs: int = COMPARISON_NUM_PAIRS,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    low_rank_dimension: int = LOREFT_DEFAULT_RANK,
    intervention_layers: str | None = None,
    learning_rate: float = COMPARISON_REFT_LEARNING_RATE,
    num_epochs: int = COMPARISON_NUM_EPOCHS_DEFAULT,
    batch_size: int = COMPARISON_TRAINING_BATCH_SIZE,
    max_length: int = COMPARISON_MAX_LENGTH,
) -> Path:
    """Train a LoReFT intervention using SFT on positive responses."""
    _original_compute_loss = pyreft.ReftTrainer.compute_loss
    def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return _original_compute_loss(self, model, inputs, return_outputs)
    pyreft.ReftTrainer.compute_loss = _patched_compute_loss

    output_path = Path(output_path)
    print(f"Step 1: Generating training data from task: {task}")
    pairs, pairs_file = generate_contrastive_pairs(task, num_pairs)
    print(f"   Loaded {len(pairs)} training examples")

    print(f"\nStep 2: Loading model {model_name}...")
    dtype = preferred_dtype(device)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if intervention_layers is None:
        layer_indices = [get_default_layer(model_name)]
    else:
        layer_indices = [int(l.strip()) for l in intervention_layers.split(",")]

    print(f"\nStep 3: Configuring LoReFT (rank={low_rank_dimension}, layers={layer_indices})...")
    hidden_size = model.config.hidden_size
    representations = []
    for layer_idx in layer_indices:
        representations.append({
            "layer": layer_idx,
            "component": "block_output",
            "low_rank_dimension": low_rank_dimension,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=hidden_size, low_rank_dimension=low_rank_dimension, dtype=dtype,
            ),
        })

    reft_config = pyreft.ReftConfig(representations=representations)
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)
    reft_model.print_trainable_parameters()

    print(f"\nStep 5: Preparing ReFT dataset...")
    prompts, responses = prepare_reft_dataset(pairs, tokenizer, max_length=max_length)
    print(f"   Dataset size: {len(prompts)} examples")
    data_module = pyreft.make_last_position_supervised_data_module(tokenizer, model, prompts, responses)

    print(f"\nStep 6: Training LoReFT intervention...")
    training_output_dir = tempfile.mkdtemp(prefix="reft_training_")
    training_args = transformers.TrainingArguments(
        output_dir=training_output_dir, num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_DEFAULT,
        learning_rate=learning_rate, weight_decay=TRAINING_WEIGHT_DECAY, warmup_ratio=TRAINING_WARMUP_RATIO,
        logging_steps=COMPARISON_LOGGING_STEPS, save_strategy="no",
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16), report_to="none",
    )
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer, args=training_args, **data_module,
    )
    trainer.train()

    print(f"\nStep 7: Saving ReFT intervention to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    reft_model.save(output_path)
    tokenizer.save_pretrained(output_path)
    metadata = {
        "method": "reft", "model": model_name, "task": task, "trait_label": trait_label,
        "num_pairs": len(pairs),
        "reft_config": {"low_rank_dimension": low_rank_dimension, "intervention_layers": layer_indices, "component": "block_output"},
        "training_config": {"learning_rate": learning_rate, "num_epochs": num_epochs, "batch_size": batch_size, "max_length": max_length},
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del reft_model, trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if not keep_intermediate:
        import os, shutil
        os.unlink(pairs_file)
        shutil.rmtree(training_output_dir, ignore_errors=True)
    print(f"\nReFT intervention saved to {output_path}")
    return output_path
