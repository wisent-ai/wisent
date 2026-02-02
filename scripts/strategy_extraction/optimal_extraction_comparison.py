#!/usr/bin/env python3
"""
Compare optimal extraction vs chat_last using raw hidden states.

This script:
1. Loads raw activations from the RawActivation database table
2. Computes initial steering direction using chat_last
3. Re-extracts at optimal positions using the optimal_extraction module
4. Compares pairwise steering accuracy between strategies

Usage:
    python -m scripts.strategy_extraction.optimal_extraction_comparison --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import psycopg2
import torch

from wisent.core.activations.core.optimal_extraction import (
    extract_at_optimal_position, extract_at_max_diff_norm, find_direction_from_all_tokens
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres?options=-c%20statement_timeout%3D0"
)


@dataclass
class RawActivationPair:
    """Container for a pair of raw activations."""
    pair_id: int
    pos_hidden_states: torch.Tensor
    neg_hidden_states: torch.Tensor
    prompt_len: int
    layer: int


def bytes_to_tensor(data: bytes, hidden_dim: int) -> torch.Tensor:
    """Convert binary data to 2D tensor [seq_len, hidden_dim]."""
    num_floats = len(data) // 4
    flat = np.array(struct.unpack(f'{num_floats}f', data))
    seq_len = num_floats // hidden_dim
    return torch.tensor(flat.reshape(seq_len, hidden_dim), dtype=torch.float32)


def count_available_data(model_name: str, benchmark: str) -> dict:
    """Count available raw activation data."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    cur.execute('''
        SELECT COUNT(DISTINCT ra."contrastivePairId"), COUNT(*), COUNT(DISTINCT ra."layer")
        FROM "RawActivation" ra
        JOIN "Model" m ON ra."modelId" = m.id
        JOIN "ContrastivePairSet" cps ON ra."contrastivePairSetId" = cps.id
        WHERE m.name = %s AND cps.name = %s
    ''', (model_name, benchmark))
    pairs, total, layers = cur.fetchone()
    cur.close()
    conn.close()
    return {"unique_pairs": pairs, "total_rows": total, "layers": layers}


def load_raw_activations(model_name: str, benchmark: str, layer: int, limit: int) -> List[RawActivationPair]:
    """Load raw activations from database."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found")
    model_id, num_layers = result
    if layer is None:
        layer = num_layers // 2
    print(f"Loading raw activations for {model_name}, layer {layer}, benchmark={benchmark}")
    cur.execute('''
        SELECT ra."contrastivePairId", ra."isPositive", ra."hiddenStates", ra."hiddenDim", ra."promptLen"
        FROM "RawActivation" ra
        JOIN "ContrastivePairSet" cps ON ra."contrastivePairSetId" = cps.id
        WHERE ra."modelId" = %s AND ra."layer" = %s AND cps.name = %s
        ORDER BY ra."contrastivePairId"
        LIMIT %s
    ''', (model_id, layer, benchmark, limit * 2))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    by_pair = defaultdict(dict)
    for row in rows:
        pair_id, is_positive, hidden_bytes, hidden_dim, prompt_len = row
        key = "pos" if is_positive else "neg"
        by_pair[pair_id][key] = {"hidden_states": bytes_to_tensor(hidden_bytes, hidden_dim), "prompt_len": prompt_len or 10}
    pairs = []
    for pair_id, data in by_pair.items():
        if "pos" in data and "neg" in data:
            pairs.append(RawActivationPair(pair_id=pair_id, pos_hidden_states=data["pos"]["hidden_states"],
                neg_hidden_states=data["neg"]["hidden_states"], prompt_len=data["pos"]["prompt_len"], layer=layer))
    print(f"Loaded {len(pairs)} complete pairs")
    return pairs


def compute_pairwise_accuracy(pos_acts: torch.Tensor, neg_acts: torch.Tensor, direction: torch.Tensor) -> float:
    """Compute pairwise steering accuracy."""
    direction = direction / (torch.norm(direction) + 1e-8)
    return (pos_acts @ direction > neg_acts @ direction).float().mean().item()


def run_comparison(model_name: str, benchmark: str, layer: int, limit: int):
    """Run the comparison between optimal and chat_last extraction."""
    print("=" * 70)
    print("OPTIMAL EXTRACTION COMPARISON")
    print("=" * 70)
    # Check available data first
    available = count_available_data(model_name, benchmark)
    print(f"Available data: {available['unique_pairs']} pairs, {available['total_rows']} rows, {available['layers']} layers")
    pairs = load_raw_activations(model_name, benchmark, layer, limit)
    if len(pairs) < 10:
        print(f"Not enough pairs ({len(pairs)}). Need at least 10.")
        return
    train_size = int(len(pairs) * 0.8)
    train_pairs, test_pairs = pairs[:train_size], pairs[train_size:]
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    train_pos_last = torch.stack([p.pos_hidden_states[-1] for p in train_pairs])
    train_neg_last = torch.stack([p.neg_hidden_states[-1] for p in train_pairs])
    chat_last_direction = train_pos_last.mean(dim=0) - train_neg_last.mean(dim=0)
    test_pos_last = torch.stack([p.pos_hidden_states[-1] for p in test_pairs])
    test_neg_last = torch.stack([p.neg_hidden_states[-1] for p in test_pairs])
    chat_last_accuracy = compute_pairwise_accuracy(test_pos_last, test_neg_last, chat_last_direction)
    print(f"\n--- CHAT_LAST ---")
    print(f"Pairwise accuracy: {chat_last_accuracy:.4f}")
    # Direction-free: extract at max ||pos - neg|| norm (no steering direction needed)
    train_pos_df, train_neg_df, train_pos_df_list = [], [], []
    for pair in train_pairs:
        result = extract_at_max_diff_norm(pair.pos_hidden_states, pair.neg_hidden_states, pair.prompt_len)
        train_pos_df.append(result.pos_activation)
        train_neg_df.append(result.neg_activation)
        train_pos_df_list.append(result.optimal_position)
    train_pos_df = torch.stack(train_pos_df)
    train_neg_df = torch.stack(train_neg_df)
    df_direction = train_pos_df.mean(dim=0) - train_neg_df.mean(dim=0)
    test_pos_df, test_neg_df, test_pos_df_list = [], [], []
    for pair in test_pairs:
        result = extract_at_max_diff_norm(pair.pos_hidden_states, pair.neg_hidden_states, pair.prompt_len)
        test_pos_df.append(result.pos_activation)
        test_neg_df.append(result.neg_activation)
        test_pos_df_list.append(result.optimal_position)
    test_pos_df = torch.stack(test_pos_df)
    test_neg_df = torch.stack(test_neg_df)
    df_accuracy = compute_pairwise_accuracy(test_pos_df, test_neg_df, df_direction)
    print(f"\n--- MAX_DIFF_NORM (direction-free) ---")
    print(f"Pairwise accuracy: {df_accuracy:.4f}")
    print(f"Positions: mean={np.mean(train_pos_df_list + test_pos_df_list):.1f}, std={np.std(train_pos_df_list + test_pos_df_list):.1f}")
    # PCA on ALL tokens: find direction without fixed position
    train_pos_hs = [p.pos_hidden_states for p in train_pairs]
    train_neg_hs = [p.neg_hidden_states for p in train_pairs]
    train_prompt_lens = [p.prompt_len for p in train_pairs]
    pca_result = find_direction_from_all_tokens(train_pos_hs, train_neg_hs, train_prompt_lens, return_details=True)
    pca_direction = pca_result.direction
    # Use PCA direction to find optimal positions, then extract
    train_pos_pca, train_neg_pca, train_pca_positions = [], [], []
    for pair in train_pairs:
        result = extract_at_optimal_position(pair.pos_hidden_states, pair.neg_hidden_states, pca_direction, pair.prompt_len)
        train_pos_pca.append(result.pos_activation)
        train_neg_pca.append(result.neg_activation)
        train_pca_positions.append(result.optimal_position)
    train_pos_pca = torch.stack(train_pos_pca)
    train_neg_pca = torch.stack(train_neg_pca)
    pca_refined_direction = train_pos_pca.mean(dim=0) - train_neg_pca.mean(dim=0)
    test_pos_pca, test_neg_pca, test_pca_positions = [], [], []
    for pair in test_pairs:
        result = extract_at_optimal_position(pair.pos_hidden_states, pair.neg_hidden_states, pca_refined_direction, pair.prompt_len)
        test_pos_pca.append(result.pos_activation)
        test_neg_pca.append(result.neg_activation)
        test_pca_positions.append(result.optimal_position)
    test_pos_pca = torch.stack(test_pos_pca)
    test_neg_pca = torch.stack(test_neg_pca)
    pca_accuracy = compute_pairwise_accuracy(test_pos_pca, test_neg_pca, pca_refined_direction)
    print(f"\n--- PCA_ALL_TOKENS (no fixed position) ---")
    print(f"Pairwise accuracy: {pca_accuracy:.4f}")
    print(f"Positions: mean={np.mean(train_pca_positions + test_pca_positions):.1f}, std={np.std(train_pca_positions + test_pca_positions):.1f}")
    print(f"PCA quality ({pca_result.n_tokens} tokens from {pca_result.n_pairs} pairs):")
    print(f"  Component 1: {pca_result.explained_variance_ratio[0]*100:.1f}% variance")
    if len(pca_result.explained_variance_ratio) > 1:
        print(f"  Component 2: {pca_result.explained_variance_ratio[1]*100:.1f}% variance")
    if len(pca_result.explained_variance_ratio) > 2:
        print(f"  Component 3: {pca_result.explained_variance_ratio[2]*100:.1f}% variance")
    top3 = sum(pca_result.explained_variance_ratio[:3]) if len(pca_result.explained_variance_ratio) >= 3 else sum(pca_result.explained_variance_ratio)
    print(f"  Top 3 total: {top3*100:.1f}% variance")
    # Two-pass optimal: use chat_last direction to find optimal positions
    train_pos_opt, train_neg_opt, train_positions = [], [], []
    for pair in train_pairs:
        result = extract_at_optimal_position(pair.pos_hidden_states, pair.neg_hidden_states, chat_last_direction, pair.prompt_len)
        train_pos_opt.append(result.pos_activation)
        train_neg_opt.append(result.neg_activation)
        train_positions.append(result.optimal_position)
    train_pos_opt = torch.stack(train_pos_opt)
    train_neg_opt = torch.stack(train_neg_opt)
    optimal_direction = train_pos_opt.mean(dim=0) - train_neg_opt.mean(dim=0)
    test_pos_opt, test_neg_opt, test_positions = [], [], []
    for pair in test_pairs:
        result = extract_at_optimal_position(pair.pos_hidden_states, pair.neg_hidden_states, optimal_direction, pair.prompt_len)
        test_pos_opt.append(result.pos_activation)
        test_neg_opt.append(result.neg_activation)
        test_positions.append(result.optimal_position)
    test_pos_opt = torch.stack(test_pos_opt)
    test_neg_opt = torch.stack(test_neg_opt)
    optimal_accuracy = compute_pairwise_accuracy(test_pos_opt, test_neg_opt, optimal_direction)
    print(f"\n--- CHAT_OPTIMAL (two-pass) ---")
    print(f"Pairwise accuracy: {optimal_accuracy:.4f}")
    print(f"Positions: mean={np.mean(train_positions + test_positions):.1f}, std={np.std(train_positions + test_positions):.1f}")
    print(f"\n--- COMPARISON ---")
    print(f"chat_last:      {chat_last_accuracy:.4f}")
    print(f"max_diff_norm:  {df_accuracy:.4f} ({(df_accuracy-chat_last_accuracy)/chat_last_accuracy*100:+.1f}%)")
    print(f"pca_all_tokens: {pca_accuracy:.4f} ({(pca_accuracy-chat_last_accuracy)/chat_last_accuracy*100:+.1f}%)")
    print(f"chat_optimal:   {optimal_accuracy:.4f} ({(optimal_accuracy-chat_last_accuracy)/chat_last_accuracy*100:+.1f}%)")
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output = {"model": model_name, "benchmark": benchmark, "layer": layer, "n_pairs": len(pairs),
        "chat_last_accuracy": chat_last_accuracy, "max_diff_norm_accuracy": df_accuracy,
        "chat_optimal_accuracy": optimal_accuracy,
        "mean_optimal_position": float(np.mean(train_positions + test_positions))}
    output_path = output_dir / f"optimal_extraction_{model_name.replace('/', '_')}_{benchmark}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare optimal vs chat_last extraction")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--benchmark", type=str, default="truthfulqa_custom", help="Benchmark")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    parser.add_argument("--limit", type=int, default=200, help="Max pairs to load")
    args = parser.parse_args()
    run_comparison(args.model, args.benchmark, args.layer, args.limit)
