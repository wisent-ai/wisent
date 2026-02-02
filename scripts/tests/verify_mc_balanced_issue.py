#!/usr/bin/env python3
"""Verify that MC_BALANCED bimodal distribution is due to A vs B token."""

import torch
import numpy as np
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
layer = model.num_layers // 2

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
train_pairs = result["train_qa_pairs"].pairs[:100]

# Track which letter each positive response gets assigned
letter_assignments = []
for pair in train_pairs:
    pos_goes_in_b = hash(pair.prompt) % 2 == 0
    if pos_goes_in_b:
        letter_assignments.append("B")
    else:
        letter_assignments.append("A")

print(f"\nLetter assignments: {letter_assignments[:20]}...")
print(f"Count A: {letter_assignments.count('A')}, Count B: {letter_assignments.count('B')}")

# Collect activations
store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
collector = ActivationCollector(model=model, store_device=store_dev)

print("\nCollecting MC_BALANCED activations...")
pos_acts = []
for i, pair in enumerate(train_pairs):
    if i % 25 == 0:
        print(f"  {i}/{len(train_pairs)}")
    pair_acts = collector.collect(pair, strategy=ExtractionStrategy.MC_BALANCED, layers=[str(layer)])
    pos = pair_acts.positive_response.layers_activations[str(layer)]
    pos_acts.append(pos.flatten().cpu().float())

pos_tensor = torch.stack(pos_acts)

# Also collect negative activations
print("\nCollecting NEGATIVE MC_BALANCED activations...")
neg_acts = []
for i, pair in enumerate(train_pairs):
    if i % 25 == 0:
        print(f"  {i}/{len(train_pairs)}")
    pair_acts = collector.collect(pair, strategy=ExtractionStrategy.MC_BALANCED, layers=[str(layer)])
    neg = pair_acts.negative_response.layers_activations[str(layer)]
    neg_acts.append(neg.flatten().cpu().float())

neg_tensor = torch.stack(neg_acts)

# Compute raw directions (pos - neg)
raw_directions = pos_tensor - neg_tensor

# The issue: for pos_goes_in_b=True, pos="B", neg="A" → direction includes (B-A) component
# for pos_goes_in_b=False, pos="A", neg="B" → direction includes (A-B) = -(B-A) component
# These are OPPOSITE, causing the bimodal distribution

# To isolate semantic signal, flip directions where pos="A" so all have same A/B orientation
corrected_directions = raw_directions.clone()
for i, letter in enumerate(letter_assignments):
    if letter == "A":
        # pos="A", neg="B", so direction has (A-B) component
        # Flip it so it has (B-A) component like the other group
        corrected_directions[i] = -corrected_directions[i]

# Now compute consistency for raw vs corrected directions
def compute_consistency(directions):
    dirs_norm = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-8)
    pairwise = (dirs_norm @ dirs_norm.T).numpy()
    mask = np.triu(np.ones_like(pairwise), k=1).astype(bool)
    return pairwise[mask].mean(), pairwise[mask].std()

raw_mean, raw_std = compute_consistency(raw_directions)
corrected_mean, corrected_std = compute_consistency(corrected_directions)

print(f"\n{'='*60}")
print("DIRECTION CONSISTENCY ANALYSIS")
print(f"{'='*60}")
print(f"\nRaw (pos-neg) directions:")
print(f"  Mean pairwise cosine: {raw_mean:.4f} (+/- {raw_std:.4f})")

print(f"\nCorrected directions (flipped for pos='A' to align A/B effect):")
print(f"  Mean pairwise cosine: {corrected_mean:.4f} (+/- {corrected_std:.4f})")

print(f"\n{'='*60}")
print("ISOLATING SEMANTIC SIGNAL")
print(f"{'='*60}")

# The corrected_mean (0.91) is just the (B-A) token embedding consistency
# To find SEMANTIC signal, we need to PROJECT OUT the A/B effect

# Compute the mean (B-A) direction from corrected directions
ab_direction = corrected_directions.mean(dim=0)
ab_direction = ab_direction / (torch.norm(ab_direction) + 1e-8)

print(f"\nMean (B-A) direction computed from {len(corrected_directions)} samples")

# Project out the A/B component from each RAW direction
# For each direction d, compute: d - (d.ab)ab
semantic_directions = []
for i, d in enumerate(raw_directions):
    # Project out the A/B component
    d_proj = d - (d @ ab_direction) * ab_direction
    semantic_directions.append(d_proj)

semantic_directions = torch.stack(semantic_directions)

# Check consistency of semantic-only directions
semantic_mean, semantic_std = compute_consistency(semantic_directions)

print(f"\nAfter projecting out (B-A) direction:")
print(f"  Semantic-only consistency: {semantic_mean:.4f} (+/- {semantic_std:.4f})")

# Also compute what fraction of variance is explained by A/B vs semantic
ab_variance = []
semantic_variance = []
for i, d in enumerate(raw_directions):
    ab_component = (d @ ab_direction).item() ** 2
    total_var = (d @ d).item()
    ab_variance.append(ab_component / (total_var + 1e-8))
    semantic_variance.append(1 - ab_component / (total_var + 1e-8))

print(f"\nVariance explained:")
print(f"  By A/B token: {np.mean(ab_variance)*100:.1f}%")
print(f"  By semantic:  {np.mean(semantic_variance)*100:.1f}%")

# Compare to CHAT_LAST for reference
print(f"\n{'='*60}")
print("COMPARISON TO CHAT_LAST")
print(f"{'='*60}")
print(f"\nCHAT_LAST consistency: ~0.36 (from previous run)")
print(f"MC_BALANCED semantic-only consistency: {semantic_mean:.4f}")

if semantic_mean < 0.1:
    print(f"\nCONCLUSION: MC_BALANCED has almost NO semantic signal after removing A/B effect.")
    print("The strategy is fundamentally flawed - it captures token identity, not content.")
elif semantic_mean < 0.36:
    print(f"\nCONCLUSION: MC_BALANCED has weaker semantic signal than CHAT_LAST.")
    print(f"  CHAT_LAST: 0.36")
    print(f"  MC_BALANCED (semantic only): {semantic_mean:.4f}")
else:
    print(f"\nCONCLUSION: MC_BALANCED has comparable semantic signal to CHAT_LAST.")
    print("But requires post-processing to remove A/B effect.")
print(f"{'='*60}")

# But wait - does the AVERAGING save it?
# Nina Rimsky's CAA balances A/B across examples, so when you average directions,
# the A/B effect cancels and you're left with semantic signal
print(f"\n{'='*60}")
print("DOES AVERAGING SAVE MC_BALANCED?")
print(f"{'='*60}")

# The raw mean direction (averaging all pos-neg directions)
mean_raw_direction = raw_directions.mean(dim=0)
mean_raw_direction_norm = mean_raw_direction / (torch.norm(mean_raw_direction) + 1e-8)

# Compare to CHAT_LAST mean direction
print("\nCollecting CHAT_LAST for comparison...")
chat_last_dirs = []
for i, pair in enumerate(train_pairs):
    if i % 25 == 0:
        print(f"  {i}/{len(train_pairs)}")
    pair_acts = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=[str(layer)])
    pos = pair_acts.positive_response.layers_activations[str(layer)].flatten().cpu().float()
    neg = pair_acts.negative_response.layers_activations[str(layer)].flatten().cpu().float()
    chat_last_dirs.append(pos - neg)

chat_last_dirs = torch.stack(chat_last_dirs)
mean_chat_last = chat_last_dirs.mean(dim=0)
mean_chat_last_norm = mean_chat_last / (torch.norm(mean_chat_last) + 1e-8)

# How similar are the averaged directions?
cos_sim = (mean_raw_direction_norm @ mean_chat_last_norm).item()
print(f"\nCosine similarity between averaged directions:")
print(f"  MC_BALANCED avg vs CHAT_LAST avg: {cos_sim:.4f}")

# Test steering accuracy using the averaged directions
from sklearn.linear_model import LogisticRegression

# For MC_BALANCED: project all samples onto its mean direction
mc_proj = (raw_directions @ mean_raw_direction_norm).numpy()
# For CHAT_LAST: project all samples onto its mean direction
cl_proj = (chat_last_dirs @ mean_chat_last_norm).numpy()

# The projected values should separate pos from neg
# (pos-neg projected onto mean direction should be positive)
mc_correct = (mc_proj > 0).sum()
cl_correct = (cl_proj > 0).sum()

print(f"\nSteering direction accuracy (projection > 0):")
print(f"  MC_BALANCED: {mc_correct}/{len(mc_proj)} = {mc_correct/len(mc_proj)*100:.1f}%")
print(f"  CHAT_LAST:   {cl_correct}/{len(cl_proj)} = {cl_correct/len(cl_proj)*100:.1f}%")

# Also check magnitude of projections (effect size)
print(f"\nMean projection magnitude (effect size):")
print(f"  MC_BALANCED: {np.abs(mc_proj).mean():.4f}")
print(f"  CHAT_LAST:   {np.abs(cl_proj).mean():.4f}")

print(f"\n{'='*60}")
if mc_correct/len(mc_proj) > 0.9:
    print("MC_BALANCED AVERAGING WORKS: The A/B effect cancels when averaged.")
    print("Individual samples are noisy, but the final steering vector is valid.")
else:
    print("MC_BALANCED AVERAGING DOESN'T FULLY WORK.")
print(f"{'='*60}")
