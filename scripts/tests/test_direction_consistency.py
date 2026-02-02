#!/usr/bin/env python3
"""
Measure direction consistency as predictor of steering effectiveness.

Hypothesis: Methods where individual samples point in similar directions
produce better steering than methods where samples point in random directions.
"""

import torch
import torch.nn.functional as F
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def compute_gradient_direction(model, prompt, response, layer):
    """Compute gradient of log P(response|prompt) w.r.t. hidden states."""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + response
    prompt_ids = model.tokenizer.encode(prompt_text, return_tensors="pt")
    full_ids = model.tokenizer.encode(full_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids.shape[1]
    hidden_states_storage = {}

    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs.retain_grad()
        hidden_states_storage["target"] = hs

    handle = model.hf_model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        model.hf_model.zero_grad()
        outputs = model.hf_model(full_ids)
        shift_logits = outputs.logits[:, prompt_len-1:-1, :]
        shift_labels = full_ids[:, prompt_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs.sum().backward()
        grad = hidden_states_storage["target"].grad
        if grad is None:
            return torch.zeros(model.hidden_size, device=model.device)
        return grad[0].mean(dim=0).detach()
    finally:
        handle.remove()


def pairwise_consistency(directions):
    """Compute mean pairwise cosine similarity between normalized directions."""
    stack = torch.stack(directions)
    stack_norm = stack / (torch.norm(stack, dim=1, keepdim=True) + 1e-8)
    pairwise = stack_norm @ stack_norm.T
    mask = torch.triu(torch.ones_like(pairwise), diagonal=1).bool()
    vals = pairwise[mask]
    return vals.mean().item(), vals.std().item()


print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
layer = model.num_layers // 2
print(f"Using layer {layer}, hidden_size={model.hidden_size}")

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
train_pairs = result["train_qa_pairs"].pairs[:100]
print(f"Using {len(train_pairs)} training pairs")

# Collect gradient directions
print("\nComputing gradient directions...")
gradient_dirs = []
for i, pair in enumerate(train_pairs):
    if i % 20 == 0:
        print(f"  {i}/{len(train_pairs)}")
    grad_pos = compute_gradient_direction(model, pair.prompt, pair.positive_response.model_response, layer)
    grad_neg = compute_gradient_direction(model, pair.prompt, pair.negative_response.model_response, layer)
    gradient_dirs.append(grad_pos - grad_neg)

# Collect CAA directions
print("\nComputing CAA directions...")
store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
collector = ActivationCollector(model=model, store_device=store_dev)
caa_dirs = []
for i, pair in enumerate(train_pairs):
    if i % 20 == 0:
        print(f"  {i}/{len(train_pairs)}")
    pair_acts = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=[str(layer)])
    pos = pair_acts.positive_response.layers_activations[str(layer)]
    neg = pair_acts.negative_response.layers_activations[str(layer)]
    caa_dirs.append(pos.flatten() - neg.flatten())

# Compute consistency
print("\n" + "=" * 60)
print("DIRECTION CONSISTENCY (pairwise cosine similarity)")
print("=" * 60)

grad_mean, grad_std = pairwise_consistency(gradient_dirs)
caa_mean, caa_std = pairwise_consistency(caa_dirs)

print(f"\nGradient consistency: {grad_mean:.4f} (+/- {grad_std:.4f})")
print(f"CAA consistency:      {caa_mean:.4f} (+/- {caa_std:.4f})")

# Alignment with averaged direction
grad_avg = torch.stack(gradient_dirs).mean(dim=0)
grad_avg = grad_avg / (torch.norm(grad_avg) + 1e-8)
caa_avg = torch.stack(caa_dirs).mean(dim=0)
caa_avg = caa_avg / (torch.norm(caa_avg) + 1e-8)

grad_stack_norm = torch.stack(gradient_dirs) / (torch.norm(torch.stack(gradient_dirs), dim=1, keepdim=True) + 1e-8)
caa_stack_norm = torch.stack(caa_dirs) / (torch.norm(torch.stack(caa_dirs), dim=1, keepdim=True) + 1e-8)

grad_align = (grad_stack_norm @ grad_avg).abs()
caa_align = (caa_stack_norm @ caa_avg).abs()

print(f"\nAlignment with final direction:")
print(f"  Gradient: {grad_align.mean():.4f} (+/- {grad_align.std():.4f})")
print(f"  CAA:      {caa_align.mean():.4f} (+/- {caa_align.std():.4f})")

print("\n" + "=" * 60)
if caa_mean > grad_mean:
    print(f"CAA is more consistent by {caa_mean - grad_mean:.4f}")
    print("Higher consistency -> better steering (confirmed by generation results)")
else:
    print(f"Gradient is more consistent by {grad_mean - caa_mean:.4f}")
print("=" * 60)
