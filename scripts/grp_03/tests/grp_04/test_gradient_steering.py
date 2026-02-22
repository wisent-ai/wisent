#!/usr/bin/env python3
"""
Test gradient-based steering vs CAA.

Instead of using mean difference (correlational), use gradient of log P(response)
w.r.t. activations (causal).
"""

import torch
import torch.nn.functional as F
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.steering_methods import CAAMethod
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def compute_gradient_direction(model: WisentModel, prompt: str, response: str, layer: int) -> torch.Tensor:
    """
    Compute gradient of log P(response|prompt) w.r.t. hidden states at layer.

    This gives us the direction that INCREASES probability of this response.
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + response

    prompt_ids = model.tokenizer.encode(prompt_text, return_tensors="pt")
    full_ids = model.tokenizer.encode(full_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids.shape[1]

    # Storage for hidden states
    hidden_states_storage = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # output is tuple (hidden_states, ...) or just hidden_states
            hs = output[0] if isinstance(output, tuple) else output
            hs.retain_grad()
            hidden_states_storage[name] = hs
        return hook_fn

    # Register hook at target layer
    layer_module = model.hf_model.model.layers[layer]
    handle = layer_module.register_forward_hook(make_hook("target"))

    try:
        # Forward pass with gradient tracking
        model.hf_model.zero_grad()
        outputs = model.hf_model(full_ids)
        logits = outputs.logits

        # Compute log prob of response tokens
        shift_logits = logits[:, prompt_len-1:-1, :]
        shift_labels = full_ids[:, prompt_len:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_log_prob = token_log_probs.sum()

        # Backward pass
        total_log_prob.backward()

        # Get gradient w.r.t. hidden states
        hidden_states = hidden_states_storage["target"]
        grad = hidden_states.grad  # [1, seq_len, hidden_dim]

        if grad is None:
            print("Warning: gradient is None")
            return torch.zeros(model.hidden_size, device=model.device)

        # Average gradient across sequence positions (or use specific position)
        # Using mean across all positions
        grad_direction = grad[0].mean(dim=0)  # [hidden_dim]

        return grad_direction.detach()

    finally:
        handle.remove()


def compute_contrastive_gradient(model: WisentModel, prompt: str, pos_response: str, neg_response: str, layer: int) -> torch.Tensor:
    """
    Compute direction that increases P(pos) and decreases P(neg).

    direction = grad(log P(pos)) - grad(log P(neg))
    """
    grad_pos = compute_gradient_direction(model, prompt, pos_response, layer)
    grad_neg = compute_gradient_direction(model, prompt, neg_response, layer)

    # Direction that increases pos probability and decreases neg probability
    direction = grad_pos - grad_neg

    return direction


print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
print(f"Model has {model.num_layers} layers, hidden_size={model.hidden_size}")

# Load data
loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
test_pairs = result["test_qa_pairs"].pairs[:20]
train_pairs = result["train_qa_pairs"]
print(f"Train: {len(train_pairs.pairs)}, Test: {len(test_pairs)}")

mid = model.num_layers // 2
layer = mid
print(f"Using layer: {layer}")

# Compute gradient-based steering direction
print("\n" + "=" * 70)
print("Computing GRADIENT-BASED steering direction...")
print("=" * 70)

gradient_directions = []
for i, pair in enumerate(train_pairs.pairs[:100]):  # Use subset for speed
    if i % 20 == 0:
        print(f"  {i}/100", flush=True)

    direction = compute_contrastive_gradient(
        model,
        pair.prompt,
        pair.positive_response.model_response,
        pair.negative_response.model_response,
        layer
    )
    gradient_directions.append(direction)

# Average directions
gradient_steering = torch.stack(gradient_directions).mean(dim=0)
gradient_steering = gradient_steering / (torch.norm(gradient_steering) + 1e-8)
print(f"Gradient steering direction norm: {torch.norm(gradient_steering):.4f}")

# Compute CAA-based steering direction for comparison
print("\n" + "=" * 70)
print("Computing CAA steering direction...")
print("=" * 70)

store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
collector = ActivationCollector(model=model, store_device=store_dev)

pairs_with_acts = []
for i, pair in enumerate(train_pairs.pairs[:100]):
    if i % 20 == 0:
        print(f"  {i}/100", flush=True)
    pair_with_acts = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=[str(layer)])
    pairs_with_acts.append(pair_with_acts)

pair_set = ContrastivePairSet(name="caa", pairs=pairs_with_acts)
caa = CAAMethod()
caa_steering = caa.train(pair_set)

# Extract direction from CAA result
caa_direction = None
for layer_key, direction in caa_steering.items():
    caa_direction = torch.tensor(direction).to(model.device)
    break

print(f"CAA steering direction norm: {torch.norm(caa_direction):.4f}")

# Compare directions
cosine_sim = F.cosine_similarity(gradient_steering.unsqueeze(0), caa_direction.unsqueeze(0).float())
print(f"\nCosine similarity between gradient and CAA directions: {cosine_sim.item():.4f}")

# Create steering plans
gradient_plan = SteeringPlan.from_raw(
    raw={str(layer): gradient_steering.cpu().numpy().tolist()},
    scale=0.5
)
caa_plan = SteeringPlan.from_raw(
    raw={str(layer): caa_direction.cpu().numpy().tolist()},
    scale=0.5
)

# Evaluate with generation
from wisent.core.evaluators.rotator import EvaluatorRotator

EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
evaluator = EvaluatorRotator(evaluator=None, task_name="truthfulqa_gen")

print("\n" + "=" * 70)
print("Evaluating with generation...")
print("=" * 70)

results = {"baseline": 0, "gradient": 0, "caa": 0}
total = 0

for i, pair in enumerate(test_pairs):
    metadata = pair.metadata or {}
    prompt_msg = [[{"role": "user", "content": pair.prompt}]]

    # Baseline
    model.detach()
    resp_base = model.generate(prompt_msg, max_new_tokens=100)[0]
    eval_base = evaluator.evaluate(
        response=resp_base,
        expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_base.ground_truth == "TRUTHFUL":
        results["baseline"] += 1

    # Gradient steering
    model.apply_steering(plan=gradient_plan)
    resp_grad = model.generate(prompt_msg, max_new_tokens=100)[0]
    model.detach()

    eval_grad = evaluator.evaluate(
        response=resp_grad,
        expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_grad.ground_truth == "TRUTHFUL":
        results["gradient"] += 1

    # CAA steering
    model.apply_steering(plan=caa_plan)
    resp_caa = model.generate(prompt_msg, max_new_tokens=100)[0]
    model.detach()

    eval_caa = evaluator.evaluate(
        response=resp_caa,
        expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_caa.ground_truth == "TRUTHFUL":
        results["caa"] += 1

    total += 1
    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(test_pairs)}] Base: {results['baseline']}, Grad: {results['gradient']}, CAA: {results['caa']}", flush=True)

# Summary
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

baseline = results["baseline"]
print(f"\n{'Method':<20} {'Correct':>10} {'Accuracy':>12} {'vs Baseline':>12}")
print("-" * 60)
print(f"{'baseline':<20} {baseline:>10} {100*baseline/total:>11.1f}% {'-':>12}")

for name in ["gradient", "caa"]:
    correct = results[name]
    delta = correct - baseline
    print(f"{name:<20} {correct:>10} {100*correct/total:>11.1f}% {delta:>+12d}")

print("=" * 70)
