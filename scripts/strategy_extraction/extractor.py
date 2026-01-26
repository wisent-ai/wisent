#!/usr/bin/env python3
"""Core extraction logic for all 7 strategies."""

import sys
import os

import torch

# Add wisent package to path - use direct import to avoid loading full geometry module
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, script_dir)

# Import directly from the extraction_strategy module to avoid geometry/numba dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extraction_strategy",
    os.path.join(script_dir, "wisent/core/activations/extraction_strategy.py")
)
extraction_strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extraction_strategy_module)

ExtractionStrategy = extraction_strategy_module.ExtractionStrategy
build_extraction_texts = extraction_strategy_module.build_extraction_texts
extract_activation = extraction_strategy_module.extract_activation

from .database import create_activation

# All 7 strategies to extract
ALL_STRATEGIES = [
    ExtractionStrategy.CHAT_MEAN,
    ExtractionStrategy.CHAT_FIRST,
    ExtractionStrategy.CHAT_LAST,
    ExtractionStrategy.CHAT_MAX_NORM,
    ExtractionStrategy.CHAT_WEIGHTED,
    ExtractionStrategy.ROLE_PLAY,
    ExtractionStrategy.MC_BALANCED,
]


def parse_pair_text(text: str) -> tuple:
    """Parse stored pair text into prompt and response.

    Format: "{prompt}\n\n{response}"
    """
    if not text:
        return "", ""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def extract_with_strategy(model, tokenizer, device, strategy: ExtractionStrategy,
                          prompt: str, response: str, other_response: str,
                          is_positive: bool, num_layers: int) -> list:
    """Extract activation for a single strategy across all layers.

    Returns list of activation tensors, one per layer.
    """
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=strategy,
        prompt=prompt,
        response=response,
        tokenizer=tokenizer,
        other_response=other_response,
        is_positive=is_positive,
        auto_convert_strategy=True,
    )

    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
    enc = {k: v.to(device) for k, v in enc.items()}

    prompt_enc = tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=2048)
    prompt_len = prompt_enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    activations = []
    for layer_idx in range(num_layers):
        hidden_states = out.hidden_states[layer_idx + 1][0]
        activation = extract_activation(
            strategy=strategy,
            hidden_states=hidden_states,
            answer_text=answer_text,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
        )
        activations.append(activation)

    return activations


def extract_pair_all_strategies(model, tokenizer, device, model_id: int,
                                pair_id: int, set_id: int, pos_text: str, neg_text: str,
                                conn, num_layers: int):
    """Extract all 7 strategies for a single pair."""
    prompt, pos_response = parse_pair_text(pos_text)
    _, neg_response = parse_pair_text(neg_text)

    # Skip chat_last since it already exists
    strategies_to_extract = [s for s in ALL_STRATEGIES if s != ExtractionStrategy.CHAT_LAST]

    for strategy in strategies_to_extract:
        pos_activations = extract_with_strategy(
            model, tokenizer, device, strategy,
            prompt=prompt,
            response=pos_response,
            other_response=neg_response,
            is_positive=True,
            num_layers=num_layers,
        )

        neg_activations = extract_with_strategy(
            model, tokenizer, device, strategy,
            prompt=prompt,
            response=neg_response,
            other_response=pos_response,
            is_positive=False,
            num_layers=num_layers,
        )

        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            create_activation(conn, model_id, pair_id, set_id, layer_num,
                              pos_activations[layer_idx], True, strategy.value)
            create_activation(conn, model_id, pair_id, set_id, layer_num,
                              neg_activations[layer_idx], False, strategy.value)

        del pos_activations, neg_activations

    if device == "cuda":
        torch.cuda.empty_cache()
