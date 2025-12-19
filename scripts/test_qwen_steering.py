#!/usr/bin/env python3
"""Test steering on Qwen3-8B with TruthfulQA generation task."""

from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.steering_methods import CAAMethod
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.extraction_strategy import ExtractionStrategy, map_legacy_strategy
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

MODEL = "Qwen/Qwen3-8B"

print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
print(f"Model has {model.num_layers} layers, hidden_size={model.hidden_size}")

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_generation", 0.8, 42, 500, None, None)
test_pairs = result["test_qa_pairs"].pairs
train_pairs = result["train_qa_pairs"]
print(f"Train: {len(train_pairs.pairs)}, Test: {len(test_pairs)}")

# Use middle layers
mid = model.num_layers // 2
layers = [str(mid-1), str(mid), str(mid+1)]
print(f"Using layers: {layers}")

print("Collecting activations...")
collector = ActivationCollector(model=model, store_device="cuda")
updated_pairs = []
for i, pair in enumerate(train_pairs.pairs):
    if i % 100 == 0:
        print(f"  Collecting {i}/{len(train_pairs.pairs)}")
    updated = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST)
    updated_pairs.append(updated)
train_with_acts = ContrastivePairSet(name="train", pairs=updated_pairs)

print("Training CAA...")
caa = CAAMethod()
steering = caa.train(train_with_acts)
plan = SteeringPlan.from_raw(raw=dict(steering), scale=0.5)

EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
evaluator = EvaluatorRotator(evaluator=None, task_name="truthfulqa_generation")

print("Evaluating...")
unsteered_correct = 0
steered_correct = 0
total = 0

for i, pair in enumerate(test_pairs[:100]):
    metadata = pair.metadata or {}
    
    resp_u = model.generate([[{"role": "user", "content": pair.prompt}]], max_new_tokens=100)[0]
    eval_u = evaluator.evaluate(response=resp_u, expected=pair.positive_response.model_response, correct_answers=metadata.get("correct_answers", []), incorrect_answers=metadata.get("incorrect_answers", []))
    if eval_u.ground_truth == "TRUTHFUL":
        unsteered_correct += 1
    
    model.apply_steering(plan)
    resp_s = model.generate([[{"role": "user", "content": pair.prompt}]], max_new_tokens=100, use_steering=True, steering_plan=plan)[0]
    model.clear_steering()
    eval_s = evaluator.evaluate(response=resp_s, expected=pair.positive_response.model_response, correct_answers=metadata.get("correct_answers", []), incorrect_answers=metadata.get("incorrect_answers", []))
    if eval_s.ground_truth == "TRUTHFUL":
        steered_correct += 1
    
    total += 1
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/100] Unsteered: {unsteered_correct}/{total} ({100*unsteered_correct/total:.1f}%), Steered: {steered_correct}/{total} ({100*steered_correct/total:.1f}%)")

print("="*60)
print(f"FINAL RESULTS - {MODEL} (100 test pairs)")
print(f"  Unsteered: {unsteered_correct}/100 = {unsteered_correct}%")
print(f"  Steered:   {steered_correct}/100 = {steered_correct}%")
print(f"  Delta:     {steered_correct-unsteered_correct:+d}%")
print("="*60)
