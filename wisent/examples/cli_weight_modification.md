# Weight Modification Examples

Permanently bake steering into model weights (no runtime overhead).
See also: [Steering Method Examples](cli_examples.md)

---

## CAA/PRISM Weight Modification

```bash
python -m wisent.core.main modify-weights \
    --task refusal \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./outputs/modified_model \
    --strength 1.0 \
    --norm-preserve \
    --verbose
```

---

## Multi-Concept Modification

Suppress multiple behaviors and/or enhance others simultaneously.

```bash
python -m wisent.core.main modify-weights \
    --task refusal \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./outputs/multi_concept \
    --concepts 'refusal:suppress:1.0' 'truthfulness:enhance:0.5' \
    --verbose
```

---

## Multi-Concept with Null-Space Constraint (AlphaEdit-style)

Suppress multiple behaviors while provably preserving others using SVD-based null-space projection.

```bash
python -m wisent.core.main modify-weights \
    --task refusal \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./outputs/null_space_modified \
    --concepts 'refusal:suppress:1.0' 'truthfulness:enhance:0.5' \
    --use-null-space \
    --verbose
```

**How it works:** Computes P_null = I - V diag(S^2/(S^2+eps)) V^T via SVD from preserved
key activations. Each weight delta is projected into the null space before application,
ensuring modifications don't affect preserved activations. Keys accumulate across concepts
so later concepts respect all prior modifications.

### Null-Space Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-null-space` | False | Enable null-space constrained editing |
| `--null-space-epsilon` | 1e-6 | Tikhonov regularization (increase for more aggressive regularization) |
| `--null-space-max-rank` | None | SVD rank truncation (set lower to reduce memory) |
| `--concepts` | - | Format: `'name:action:strength'` (actions: suppress, enhance) |
| `--no-orthogonalize` | False | Disable Gram-Schmidt orthogonalization |

### Python API

```python
from wisent.core.weight_modification import (
    PreservedKeyMatrix, run_multi_concept_modification,
    MultiConceptConfig, ConceptSpec, ConceptAction,
)

# Build preserved keys from harmless activations
preserved = PreservedKeyMatrix(epsilon=1e-6)
preserved.accumulate(harmless_vectors)  # keys to protect

config = MultiConceptConfig(
    use_null_space=True,
    null_space_epsilon=1e-6,
    accumulate_keys_across_concepts=True,
)

result = run_multi_concept_modification(
    model, concepts=[
        ConceptSpec("refusal", steering_vecs, ConceptAction.SUPPRESS, strength=1.0),
        ConceptSpec("truthfulness", truth_vecs, ConceptAction.ENHANCE, strength=0.5),
    ],
    config=config,
    preserved_keys=preserved,
)
```

---

## TITAN Hybrid Mode (Weights + Runtime Hooks)

```python
from wisent.core.weight_modification import apply_titan_steering

result = apply_titan_steering(
    model=model,
    titan_result=titan_result,
    mode="hybrid",  # "static", "dynamic", or "hybrid"
    base_strength=1.0,
)

hooks = result["hooks"]
print(f"Gate value: {hooks.get_current_gate()}")
output = model.generate(...)
hooks.remove()
```

---

## Quick Reference

| Method | Speed | Expressiveness | Best For |
|--------|-------|----------------|----------|
| **CAA** | Fast | Low | Simple behaviors, quick experiments |
| **PRISM** | Medium | Medium | Complex behaviors needing multiple directions |
| **PULSE** | Medium | Medium-High | Context-dependent steering |
| **TITAN** | Slow | High | Maximum control, production deployment |
| **Null-Space** | Medium | High | Multi-concept with provable preservation |

### Recommended Workflows

**Quick Prototyping:**
```bash
python -m wisent.core.main train-steering MODEL --steering-method CAA --pairs PAIRS
```

**Multi-Concept with Preservation:**
```bash
python -m wisent.core.main modify-weights --task refusal --model MODEL --output-dir OUT \
    --concepts 'refusal:suppress:1.0' 'coding:enhance:0.5' --use-null-space --verbose
```

**Production Deployment:**
```bash
python -m wisent.core.main train-steering MODEL --steering-method TITAN --titan-optimization-steps 300 --pairs PAIRS
```
