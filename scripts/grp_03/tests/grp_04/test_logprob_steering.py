#!/usr/bin/env python3
"""
Test steering effectiveness via log probability shifts.

Compares extraction strategies: CHAT_LAST, ROLE_PLAY, MC_BALANCED, and optimal.
"""

import torch
import torch.nn.functional as F
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.steering_methods import CAAMethod
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def compute_response_logprob(model: WisentModel, prompt: str, response: str) -> float:
    """Compute log probability of response given prompt."""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + response

    prompt_ids = model.tokenizer.encode(prompt_text, return_tensors="pt")
    full_ids = model.tokenizer.encode(full_text, return_tensors="pt")

    prompt_len = prompt_ids.shape[1]
    full_ids = full_ids.to(model.device)

    with torch.no_grad():
        outputs = model.hf_model(full_ids)
        logits = outputs.logits

    shift_logits = logits[:, prompt_len-1:-1, :]
    shift_labels = full_ids[:, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()


print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
print(f"Model has {model.num_layers} layers, hidden_size={model.hidden_size}")

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
test_pairs = result["test_qa_pairs"].pairs[:20]
train_pairs = result["train_qa_pairs"]
print(f"Train: {len(train_pairs.pairs)}, Test: {len(test_pairs)}")

mid = model.num_layers // 2
layer = str(mid)
print(f"Using layer: {layer}")

store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
collector = ActivationCollector(model=model, store_device=store_dev)

strategies = [
    ExtractionStrategy.CHAT_LAST,
    ExtractionStrategy.ROLE_PLAY,
    ExtractionStrategy.MC_BALANCED,
]

steering_plans = {}

for strategy in strategies:
    print(f"\nCollecting activations with {strategy.value}...")
    pairs_with_acts = []

    for i, pair in enumerate(train_pairs.pairs):
        if i % 100 == 0:
            print(f"  {i}/{len(train_pairs.pairs)}", flush=True)

        pair_with_acts = collector.collect(pair, strategy=strategy, layers=[layer])
        pairs_with_acts.append(pair_with_acts)

    pair_set = ContrastivePairSet(name=strategy.value, pairs=pairs_with_acts)

    print(f"  Training CAA for {strategy.value}...")
    caa = CAAMethod()
    steering = caa.train(pair_set)
    plan = SteeringPlan.from_raw(raw=dict(steering), scale=0.5)
    steering_plans[strategy.value] = plan

# Evaluate via generation
from wisent.core.evaluators.rotator import EvaluatorRotator

EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
evaluator = EvaluatorRotator(evaluator=None, task_name="truthfulqa_gen")

print("\n" + "=" * 70)
print("Evaluating with actual generation...")
print("=" * 70)

gen_results = {"baseline": 0}
for name in steering_plans.keys():
    gen_results[name] = 0

total = 0
for i, pair in enumerate(test_pairs):
    metadata = pair.metadata or {}
    prompt_msg = [[{"role": "user", "content": pair.prompt}]]

    # Baseline
    model.detach()
    resp_base = model.generate(prompt_msg, max_new_tokens=100)[0]
    eval_base = evaluator.evaluate(
        response=resp_base, expected=pair.positive_response.model_response,
        correct_answers=metadata.get("correct_answers", []),
        incorrect_answers=metadata.get("incorrect_answers", [])
    )
    if eval_base.ground_truth == "TRUTHFUL":
        gen_results["baseline"] += 1

    # Each strategy
    for name, plan in steering_plans.items():
        model.apply_steering(plan=plan)
        resp = model.generate(prompt_msg, max_new_tokens=100)[0]
        model.detach()

        eval_res = evaluator.evaluate(
            response=resp, expected=pair.positive_response.model_response,
            correct_answers=metadata.get("correct_answers", []),
            incorrect_answers=metadata.get("incorrect_answers", [])
        )
        if eval_res.ground_truth == "TRUTHFUL":
            gen_results[name] += 1

    total += 1
    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(test_pairs)}] Base: {gen_results['baseline']}/{total}", flush=True)

# Summarize
print("\n" + "=" * 70)
print("GENERATION RESULTS")
print("=" * 70)

baseline = gen_results["baseline"]
print(f"\n{'Strategy':<20} {'Correct':>10} {'Accuracy':>12} {'vs Baseline':>12}")
print("-" * 60)
print(f"{'baseline':<20} {baseline:>10} {100*baseline/total:>11.1f}% {'-':>12}")

for name in steering_plans.keys():
    correct = gen_results[name]
    delta = correct - baseline
    print(f"{name:<20} {correct:>10} {100*correct/total:>11.1f}% {delta:>+12d}")

print("=" * 70)
