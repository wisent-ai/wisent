#!/usr/bin/env python3
"""Optimized extraction logic - shares forward passes across strategies."""

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

from .database import create_activations_batch

# Strategies grouped by prompt format (same format = same forward pass)
# All 5 chat strategies share the same prompt format
CHAT_STRATEGIES = [
    ExtractionStrategy.CHAT_MEAN,
    ExtractionStrategy.CHAT_FIRST,
    ExtractionStrategy.CHAT_LAST,  # Include all strategies - pairs might have 0
    ExtractionStrategy.CHAT_MAX_NORM,
    ExtractionStrategy.CHAT_WEIGHTED,
]

# These each need their own forward pass (different prompt format)
ROLE_PLAY_STRATEGY = ExtractionStrategy.ROLE_PLAY
MC_BALANCED_STRATEGY = ExtractionStrategy.MC_BALANCED


def parse_pair_text(text: str) -> tuple:
    """Parse stored pair text into prompt and response."""
    if not text:
        return "", ""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def run_model_forward(model, tokenizer, device, full_text: str, prompt_only: str):
    """Run model forward pass and return hidden states + prompt length."""
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
    enc = {k: v.to(device) for k, v in enc.items()}

    prompt_enc = tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=2048)
    prompt_len = prompt_enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    return out.hidden_states, prompt_len


def extract_all_strategies_from_hidden_states(
    strategies: list, hidden_states_tuple, answer_text: str, tokenizer, prompt_len: int, num_layers: int
) -> dict:
    """Extract activations for multiple strategies from the same hidden states."""
    results = {}
    for strategy in strategies:
        activations = []
        for layer_idx in range(num_layers):
            hidden_states = hidden_states_tuple[layer_idx + 1][0]
            activation = extract_activation(
                strategy=strategy,
                hidden_states=hidden_states,
                answer_text=answer_text,
                tokenizer=tokenizer,
                prompt_len=prompt_len,
            )
            activations.append(activation)
        results[strategy] = activations
    return results


def extract_pair_all_strategies(model, tokenizer, device, model_id: int,
                                pair_id: int, set_id: int, pos_text: str, neg_text: str,
                                conn, num_layers: int):
    """Extract all 7 strategies for a single pair (optimized).

    Uses ON CONFLICT DO NOTHING, so existing strategies are safely skipped.
    """
    prompt, pos_response = parse_pair_text(pos_text)
    _, neg_response = parse_pair_text(neg_text)

    # Collect all activations to batch insert
    batch_records = []

    # === 1. CHAT FORMAT: One forward pass, extract 4 strategies ===
    # Build chat format text (same for all chat_* strategies)
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=ExtractionStrategy.CHAT_LAST,  # Any chat strategy, same format
        prompt=prompt,
        response=pos_response,
        tokenizer=tokenizer,
        auto_convert_strategy=True,
    )
    hidden_states, prompt_len = run_model_forward(model, tokenizer, device, full_text, prompt_only)
    pos_chat_activations = extract_all_strategies_from_hidden_states(
        CHAT_STRATEGIES, hidden_states, answer_text, tokenizer, prompt_len, num_layers
    )
    del hidden_states

    # Negative for chat format
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=ExtractionStrategy.CHAT_LAST,
        prompt=prompt,
        response=neg_response,
        tokenizer=tokenizer,
        auto_convert_strategy=True,
    )
    hidden_states, prompt_len = run_model_forward(model, tokenizer, device, full_text, prompt_only)
    neg_chat_activations = extract_all_strategies_from_hidden_states(
        CHAT_STRATEGIES, hidden_states, answer_text, tokenizer, prompt_len, num_layers
    )
    del hidden_states

    # Add chat activations to batch
    for strategy in CHAT_STRATEGIES:
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            batch_records.append((model_id, pair_id, set_id, layer_num,
                                  pos_chat_activations[strategy][layer_idx], True, strategy.value))
            batch_records.append((model_id, pair_id, set_id, layer_num,
                                  neg_chat_activations[strategy][layer_idx], False, strategy.value))
    del pos_chat_activations, neg_chat_activations

    # === 2. ROLE_PLAY FORMAT: One forward pass ===
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=ROLE_PLAY_STRATEGY,
        prompt=prompt,
        response=pos_response,
        tokenizer=tokenizer,
        other_response=neg_response,
        is_positive=True,
        auto_convert_strategy=True,
    )
    hidden_states, prompt_len = run_model_forward(model, tokenizer, device, full_text, prompt_only)
    for layer_idx in range(num_layers):
        activation = extract_activation(
            strategy=ROLE_PLAY_STRATEGY,
            hidden_states=hidden_states[layer_idx + 1][0],
            answer_text=answer_text,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
        )
        batch_records.append((model_id, pair_id, set_id, layer_idx + 1,
                              activation, True, ROLE_PLAY_STRATEGY.value))
    del hidden_states

    # Negative for role_play
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=ROLE_PLAY_STRATEGY,
        prompt=prompt,
        response=neg_response,
        tokenizer=tokenizer,
        other_response=pos_response,
        is_positive=False,
        auto_convert_strategy=True,
    )
    hidden_states, prompt_len = run_model_forward(model, tokenizer, device, full_text, prompt_only)
    for layer_idx in range(num_layers):
        activation = extract_activation(
            strategy=ROLE_PLAY_STRATEGY,
            hidden_states=hidden_states[layer_idx + 1][0],
            answer_text=answer_text,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
        )
        batch_records.append((model_id, pair_id, set_id, layer_idx + 1,
                              activation, False, ROLE_PLAY_STRATEGY.value))
    del hidden_states

    # === 3. MC_BALANCED FORMAT: One forward pass ===
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=MC_BALANCED_STRATEGY,
        prompt=prompt,
        response=pos_response,
        tokenizer=tokenizer,
        other_response=neg_response,
        is_positive=True,
        auto_convert_strategy=True,
    )
    hidden_states, prompt_len = run_model_forward(model, tokenizer, device, full_text, prompt_only)
    for layer_idx in range(num_layers):
        activation = extract_activation(
            strategy=MC_BALANCED_STRATEGY,
            hidden_states=hidden_states[layer_idx + 1][0],
            answer_text=answer_text,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
        )
        batch_records.append((model_id, pair_id, set_id, layer_idx + 1,
                              activation, True, MC_BALANCED_STRATEGY.value))
    del hidden_states

    # Negative for mc_balanced
    full_text, answer_text, prompt_only = build_extraction_texts(
        strategy=MC_BALANCED_STRATEGY,
        prompt=prompt,
        response=neg_response,
        tokenizer=tokenizer,
        other_response=pos_response,
        is_positive=False,
        auto_convert_strategy=True,
    )
    hidden_states, prompt_len = run_model_forward(model, tokenizer, device, full_text, prompt_only)
    for layer_idx in range(num_layers):
        activation = extract_activation(
            strategy=MC_BALANCED_STRATEGY,
            hidden_states=hidden_states[layer_idx + 1][0],
            answer_text=answer_text,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
        )
        batch_records.append((model_id, pair_id, set_id, layer_idx + 1,
                              activation, False, MC_BALANCED_STRATEGY.value))
    del hidden_states

    # === BATCH INSERT all activations at once ===
    create_activations_batch(conn, batch_records)

    if device == "cuda":
        torch.cuda.empty_cache()
