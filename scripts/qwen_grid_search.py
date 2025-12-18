#!/usr/bin/env python3
"""Grid search for optimal Qwen3-8B steering parameters."""

from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.steering_methods import CAAMethod
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.extraction_strategy import ExtractionStrategy, map_legacy_strategy
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

print("Loading Qwen/Qwen3-8B...")
model = WisentModel("Qwen/Qwen3-8B")
num_layers = model.num_layers
print(f"Model has {num_layers} layers")

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_generation", 0.8, 42, 500, None, None)
test_pairs = result["test_qa_pairs"].pairs
train_pairs = result["train_qa_pairs"]
print(f"Train: {len(train_pairs.pairs)}, Test: {len(test_pairs)}")

EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
evaluator = EvaluatorRotator(evaluator=None, task_name="truthfulqa_generation")

# Test different layer configs for 36-layer model
configs = [
    (["8","9","10"], 0.3),   # Early layers (~25%)
    (["17","18","19"], 0.3), # Mid layers (~50%)  
    (["26","27","28"], 0.3), # Late layers (~75%)
    (["17","18","19"], 0.5), # Mid stronger
    (["17","18","19"], 0.1), # Mid weaker
]

print("\n" + "="*60)
print("GRID SEARCH: Testing 5 configurations")
print("="*60)

results = []
for layers, strength in configs:
    print(f"\nConfig: layers={layers}, strength={strength}")
    collector = ActivationCollector(model=model, store_device="cuda")
    updated = [collector.collect(p, strategy=ExtractionStrategy.CHAT_LAST) for p in train_pairs.pairs]
    train_with_acts = ContrastivePairSet(name="train", pairs=updated)
    caa = CAAMethod()
    steering = caa.train(train_with_acts)
    plan = SteeringPlan.from_raw(raw=dict(steering), scale=strength)
    
    unsteered = steered = 0
    for p in test_pairs[:50]:
        metadata = p.metadata or {}
        resp_u = model.generate([[{"role": "user", "content": p.prompt}]], max_new_tokens=100)[0]
        if evaluator.evaluate(response=resp_u, expected=p.positive_response.model_response, correct_answers=metadata.get("correct_answers", []), incorrect_answers=metadata.get("incorrect_answers", [])).ground_truth == "TRUTHFUL":
            unsteered += 1
        model.apply_steering(plan)
        resp_s = model.generate([[{"role": "user", "content": p.prompt}]], max_new_tokens=100, use_steering=True, steering_plan=plan)[0]
        model.clear_steering()
        if evaluator.evaluate(response=resp_s, expected=p.positive_response.model_response, correct_answers=metadata.get("correct_answers", []), incorrect_answers=metadata.get("incorrect_answers", [])).ground_truth == "TRUTHFUL":
            steered += 1
    
    delta = (steered - unsteered) * 2
    results.append((layers, strength, unsteered*2, steered*2, delta))
    print(f"  Unsteered: {unsteered}/50 = {unsteered*2}%, Steered: {steered}/50 = {steered*2}%, Delta: {delta:+d}%")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for layers, strength, u, s, d in results:
    print(f"  layers={layers}, strength={strength}: {u}% -> {s}% (delta={d:+d}%)")

best = max(results, key=lambda x: x[4])
print(f"\nBest config: layers={best[0]}, strength={best[1]} with delta={best[4]:+d}%")
print("\nDone!")
