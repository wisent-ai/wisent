"""Data loading and activation extraction for mixed concept detection."""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset

from wisent.core.models.wisent_model import WisentModel
from wisent.core.constants import TOKENIZER_MAX_LENGTH_GEOMETRY, DEFAULT_RANDOM_SEED, N_BOOTSTRAP_DEFAULT, PROGRESS_LOG_INTERVAL_20


def load_truthfulqa_pairs(n_pairs: int = N_BOOTSTRAP_DEFAULT, seed: int = DEFAULT_RANDOM_SEED) -> List[Dict]:
    """Load contrastive pairs from TruthfulQA."""
    random.seed(seed)
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    pairs = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    for idx in indices[:n_pairs]:
        sample = ds[idx]
        if sample["incorrect_answers"]:
            pairs.append({
                "question": sample["question"],
                "positive": sample["best_answer"],
                "negative": random.choice(sample["incorrect_answers"]),
                "source": "truthfulqa",
            })
    
    return pairs


def load_hellaswag_pairs(n_pairs: int = N_BOOTSTRAP_DEFAULT, seed: int = DEFAULT_RANDOM_SEED) -> List[Dict]:
    """Load contrastive pairs from HellaSwag."""
    random.seed(seed)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    
    pairs = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    for idx in indices[:n_pairs]:
        sample = ds[idx]
        correct_idx = int(sample["label"])
        endings = sample["endings"]
        
        # Get incorrect endings
        incorrect_indices = [i for i in range(len(endings)) if i != correct_idx]
        if incorrect_indices:
            incorrect_idx = random.choice(incorrect_indices)
            
            context = sample["ctx"]
            pairs.append({
                "question": context,
                "positive": endings[correct_idx],
                "negative": endings[incorrect_idx],
                "source": "hellaswag",
            })
    
    return pairs[:n_pairs]


def get_activations(model: WisentModel, text: str, layer: int) -> torch.Tensor:
    """Extract last token activation from a specific layer."""
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LENGTH_GEOMETRY)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    activations = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0][:, -1, :].detach().cpu()
        else:
            activations["hidden"] = output[:, -1, :].detach().cpu()
    
    layers = model._layers
    handle = layers[layer].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model.hf_model(**inputs)
    
    handle.remove()
    return activations["hidden"].squeeze(0)


def extract_difference_vectors(
    model: WisentModel,
    pairs: List[Dict],
    layer: int,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract difference vectors (positive - negative) for all pairs.
    
    Returns:
        diff_vectors: [N, hidden_dim] array of difference vectors
        sources: list of source labels (for validation only, not used in detection)
    """
    diff_vectors = []
    sources = []
    
    total = len(pairs)
    for i, pair in enumerate(pairs):
        if show_progress and (i + 1) % PROGRESS_LOG_INTERVAL_20 == 0:
            print(f"  Extracting activations: {i+1}/{total}")
        
        # Format as chat
        prompt = pair["question"]
        
        pos_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["positive"]}],
            tokenize=False, add_generation_prompt=False
        )
        neg_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["negative"]}],
            tokenize=False, add_generation_prompt=False
        )
        
        pos_act = get_activations(model, pos_text, layer)
        neg_act = get_activations(model, neg_text, layer)
        
        diff = (pos_act - neg_act).numpy()
        diff_vectors.append(diff)
        sources.append(pair.get("source", "unknown"))
    
    return np.array(diff_vectors), sources
