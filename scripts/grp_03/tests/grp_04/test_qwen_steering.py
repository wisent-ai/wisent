#!/usr/bin/env python3
"""Compare chat_last vs optimal extraction steering on Qwen3-8B with TruthfulQA."""

import torch
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.steering_methods import CAAMethod
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.activations.core.optimal_extraction import (
    find_direction_from_all_tokens,
    extract_at_optimal_position,
)

MODEL = "Qwen/Qwen3-8B"

print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
print(f"Model has {model.num_layers} layers, hidden_size={model.hidden_size}")

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
test_pairs = result["test_qa_pairs"].pairs
train_pairs = result["train_qa_pairs"]
print(f"Train: {len(train_pairs.pairs)}, Test: {len(test_pairs)}")

# Use middle layer only for simplicity
mid = model.num_layers // 2
layer = str(mid)
print(f"Using layer: {layer}")

store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
collector = ActivationCollector(model=model, store_device=store_dev)

# === COLLECT RAW ACTIVATIONS ===
print("Collecting raw activations...")
raw_data = []
for i, pair in enumerate(train_pairs.pairs):
    if i % 100 == 0:
        print(f"  Collecting {i}/{len(train_pairs.pairs)}", flush=True)
    raw = collector.collect_raw(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=[layer])
    raw_data.append({
        "pair": pair,
        "pos_hs": raw["pos_hidden_states"][layer],
        "neg_hs": raw["neg_hidden_states"][layer],
        "pos_prompt_len": raw["pos_prompt_len"],
        "neg_prompt_len": raw["neg_prompt_len"],
    })

# === CHAT_LAST EXTRACTION ===
print("Extracting with CHAT_LAST...")
chat_last_pairs = []
for rd in raw_data:
    pos_act = rd["pos_hs"][-1]  # Last token
    neg_act = rd["neg_hs"][-1]
    pair = rd["pair"].with_activations(
        positive=LayerActivations({layer: pos_act}),
        negative=LayerActivations({layer: neg_act}),
    )
    chat_last_pairs.append(pair)
chat_last_set = ContrastivePairSet(name="chat_last", pairs=chat_last_pairs)

# === OPTIMAL EXTRACTION (PCA + optimal position) ===
print("Extracting with OPTIMAL (PCA direction)...")
pos_batch = [rd["pos_hs"] for rd in raw_data]
neg_batch = [rd["neg_hs"] for rd in raw_data]
prompt_lens = [rd["pos_prompt_len"] for rd in raw_data]

# Find direction using PCA on all tokens (no initial direction needed)
pca_direction = find_direction_from_all_tokens(pos_batch, neg_batch, prompt_lens)
print(f"  PCA direction norm: {pca_direction.norm().item():.4f}")

# Extract at optimal positions using PCA direction
optimal_pairs = []
positions = []
for rd in raw_data:
    result = extract_at_optimal_position(
        rd["pos_hs"], rd["neg_hs"], pca_direction, rd["pos_prompt_len"]
    )
    pair = rd["pair"].with_activations(
        positive=LayerActivations({layer: result.pos_activation}),
        negative=LayerActivations({layer: result.neg_activation}),
    )
    optimal_pairs.append(pair)
    positions.append(result.optimal_position)
optimal_set = ContrastivePairSet(name="optimal", pairs=optimal_pairs)
print(f"  Mean optimal position: {sum(positions)/len(positions):.1f}")

# === TRAIN STEERING VECTORS ===
print("Training CAA for chat_last...")
caa = CAAMethod()
chat_last_steering = caa.train(chat_last_set)
chat_last_plan = SteeringPlan.from_raw(raw=dict(chat_last_steering), scale=0.5)

print("Training CAA for optimal...")
optimal_steering = caa.train(optimal_set)
optimal_plan = SteeringPlan.from_raw(raw=dict(optimal_steering), scale=0.5)

# === EVALUATE ===
EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
evaluator = EvaluatorRotator(evaluator=None, task_name="truthfulqa_gen")

print("Evaluating (baseline, chat_last steered, optimal steered)...")
baseline_correct = 0
chat_last_correct = 0
optimal_correct = 0
total = 0

for i, pair in enumerate(test_pairs[:100]):
    metadata = pair.metadata or {}
    prompt_msg = [[{"role": "user", "content": pair.prompt}]]

    # Baseline (unsteered)
    resp_base = model.generate(prompt_msg, max_new_tokens=100)[0]
    eval_base = evaluator.evaluate(
        response=resp_base, expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_base.ground_truth == "TRUTHFUL":
        baseline_correct += 1

    # Chat_last steered
    resp_cl = model.generate(prompt_msg, max_new_tokens=100, use_steering=True, steering_plan=chat_last_plan)[0]
    eval_cl = evaluator.evaluate(
        response=resp_cl, expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_cl.ground_truth == "TRUTHFUL":
        chat_last_correct += 1

    # Optimal steered
    resp_opt = model.generate(prompt_msg, max_new_tokens=100, use_steering=True, steering_plan=optimal_plan)[0]
    eval_opt = evaluator.evaluate(
        response=resp_opt, expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_opt.ground_truth == "TRUTHFUL":
        optimal_correct += 1

    total += 1
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/100] Base: {baseline_correct}/{total} ({100*baseline_correct/total:.1f}%), "
              f"ChatLast: {chat_last_correct}/{total} ({100*chat_last_correct/total:.1f}%), "
              f"Optimal: {optimal_correct}/{total} ({100*optimal_correct/total:.1f}%)", flush=True)

print("="*70)
print(f"FINAL RESULTS - {MODEL} (100 test pairs)")
print(f"  Baseline:   {baseline_correct}/100 = {baseline_correct}%")
print(f"  Chat_last:  {chat_last_correct}/100 = {chat_last_correct}% (delta: {chat_last_correct-baseline_correct:+d}%)")
print(f"  Optimal:    {optimal_correct}/100 = {optimal_correct}% (delta: {optimal_correct-baseline_correct:+d}%)")
print(f"  Optimal vs Chat_last: {optimal_correct-chat_last_correct:+d}%")
print("="*70)
