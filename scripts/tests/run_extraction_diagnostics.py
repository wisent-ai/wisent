#!/usr/bin/env python3
"""
Run extraction strategy diagnostics using the new consolidated module.

Demonstrates usage of wisent.core.activations.core.diagnostics API.
"""

from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.core.diagnostics import run_full_diagnostics

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
N_SAMPLES = 50  # Use fewer samples for faster testing

print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
layer = model.num_layers // 2
print(f"Using layer {layer}")

print("\nLoading contrastive pairs...")
loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
pairs = result["train_qa_pairs"].pairs[:N_SAMPLES]
print(f"Using {len(pairs)} pairs")

print("\nRunning full diagnostics...")
diagnostics, report = run_full_diagnostics(model, pairs, layer)

print("\n" + report)

# Show top 3 strategies
print("\nTop 3 strategies by overall score:")
for i, d in enumerate(sorted(diagnostics, key=lambda x: x.overall_score, reverse=True)[:3]):
    print(f"  {i+1}. {d.strategy}: {d.overall_score:.3f}")
