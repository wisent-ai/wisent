# Wisent CLI Examples

This document provides CLI examples for all steering methods available in Wisent.

## Table of Contents
- [Basic Steering (CAA)](#basic-steering-caa)
- [PRISM - Multi-Directional Steering](#prism---multi-directional-steering)
- [PULSE - Conditional Layer-Adaptive Steering](#pulse---conditional-layer-adaptive-steering)
- [TITAN - Joint Optimization Steering](#titan---joint-optimization-steering)
- [Optimization Commands](#optimization-commands)
- [Weight Modification](#weight-modification)

---

## Basic Steering (CAA)

Contrastive Activation Addition - the simplest and fastest steering method.

### Generate Contrastive Pairs
```bash
python -m wisent.core.main generate-pairs \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa \
    --output ./outputs/pairs.json \
    --limit 100 \
    --verbose
```

### Train CAA Steering Vector
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/vectors \
    --steering-method CAA \
    --caa-normalize \
    --verbose
```

### Generate Steered Responses
```bash
python -m wisent.core.main generate-responses \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa \
    --vector ./outputs/vectors/steering_vector.pt \
    --layer 12 \
    --strength 1.5 \
    --output ./outputs/responses.json \
    --num-questions 20 \
    --verbose
```

---

## PRISM - Multi-Directional Steering

PRISM (Projected Representations for Independent Steering Manifolds) discovers multiple 
steering directions per layer using gradient optimization. Best for complex behaviors 
that can't be captured by a single direction.

### Basic PRISM Training
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/prism_vectors \
    --steering-method PRISM \
    --prism-num-directions 3 \
    --prism-optimization-steps 100 \
    --prism-learning-rate 0.01 \
    --verbose
```

### PRISM with Full Configuration
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/prism_vectors \
    --steering-method PRISM \
    --prism-num-directions 5 \
    --prism-optimization-steps 200 \
    --prism-learning-rate 0.01 \
    --prism-retain-weight 0.1 \
    --prism-independence-weight 0.05 \
    --prism-use-caa-init \
    --prism-cone-constraint \
    --prism-min-cosine-similarity 0.3 \
    --prism-max-cosine-similarity 0.95 \
    --prism-normalize \
    --verbose
```

### PRISM Parameter Guide
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prism-num-directions` | 3 | Number of directions per layer (more = more expressive) |
| `--prism-optimization-steps` | 100 | Gradient steps (more = better convergence) |
| `--prism-learning-rate` | 0.01 | Step size for optimization |
| `--prism-retain-weight` | 0.1 | Preserve behavior on non-target examples |
| `--prism-independence-weight` | 0.05 | Encourage diverse directions |
| `--prism-use-caa-init` | True | Initialize with CAA direction |
| `--prism-cone-constraint` | True | Keep directions in same half-space |

---

## PULSE - Conditional Layer-Adaptive Steering

PULSE (Probabilistic Uncertainty-guided Layer Steering Engine) applies steering 
conditionally based on input content, with learned per-layer scaling.

### Basic PULSE Training
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/pulse_vectors \
    --steering-method PULSE \
    --pulse-sensor-layer 15 \
    --pulse-steering-layers "12,13,14,15,16,17,18" \
    --pulse-condition-threshold 0.5 \
    --verbose
```

### PULSE with Entropy-Based Intensity
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/pulse_vectors \
    --steering-method PULSE \
    --pulse-sensor-layer 15 \
    --pulse-steering-layers "12,13,14,15,16,17,18" \
    --pulse-per-layer-scaling \
    --pulse-condition-threshold 0.5 \
    --pulse-gate-temperature 0.1 \
    --pulse-learn-threshold \
    --pulse-use-entropy-scaling \
    --pulse-entropy-floor 0.5 \
    --pulse-entropy-ceiling 2.0 \
    --pulse-max-alpha 2.0 \
    --pulse-optimization-steps 100 \
    --pulse-learning-rate 0.01 \
    --verbose
```

### PULSE Parameter Guide
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pulse-sensor-layer` | 15 | Layer for gating decision |
| `--pulse-steering-layers` | "12-18" | Layers to apply steering |
| `--pulse-condition-threshold` | 0.5 | Activation threshold (0-1) |
| `--pulse-gate-temperature` | 0.1 | Sigmoid sharpness (lower = sharper) |
| `--pulse-per-layer-scaling` | True | Learn layer-specific strengths |
| `--pulse-use-entropy-scaling` | True | Modulate by model uncertainty |
| `--pulse-max-alpha` | 2.0 | Maximum steering intensity |

---

## TITAN - Joint Optimization Steering

TITAN (Total Integrated Targeted Activation Navigation) is the most powerful method,
jointly optimizing:
- Multi-directional manifold discovery
- Learned gating network (when to steer)
- Per-input intensity prediction
- Direction weighting within manifold

### Basic TITAN Training
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/titan_vectors \
    --steering-method TITAN \
    --titan-num-directions 5 \
    --titan-optimization-steps 200 \
    --verbose
```

### TITAN with Full Configuration
```bash
python -m wisent.core.main train-steering \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/pairs.json \
    --output ./outputs/titan_vectors \
    --steering-method TITAN \
    --titan-num-directions 5 \
    --titan-steering-layers "10,11,12,13,14,15,16,17,18" \
    --titan-sensor-layer 15 \
    --titan-gate-hidden-dim 128 \
    --titan-intensity-hidden-dim 64 \
    --titan-optimization-steps 200 \
    --titan-learning-rate 0.005 \
    --titan-behavior-weight 1.0 \
    --titan-retain-weight 0.2 \
    --titan-sparse-weight 0.05 \
    --titan-max-alpha 3.0 \
    --titan-normalize \
    --verbose
```

### TITAN Parameter Guide
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--titan-num-directions` | 5 | Directions in steering manifold |
| `--titan-steering-layers` | "10-18" | Layers for steering |
| `--titan-sensor-layer` | 15 | Primary gating layer |
| `--titan-gate-hidden-dim` | 128 | Gating network size |
| `--titan-intensity-hidden-dim` | 64 | Intensity network size |
| `--titan-optimization-steps` | 200 | Joint optimization steps |
| `--titan-behavior-weight` | 1.0 | Effectiveness loss weight |
| `--titan-retain-weight` | 0.2 | Side-effect minimization |
| `--titan-sparse-weight` | 0.05 | Layer sparsity loss |
| `--titan-max-alpha` | 3.0 | Maximum intensity |

---

## Optimization Commands

The `optimize-steering` command now supports all steering methods (CAA, PRISM, PULSE, TITAN).
Each method's parameters can be configured via CLI flags.

### Optimize Steering Parameters (Any Method)
```bash
# CAA optimization (baseline)
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods CAA \
    --limit 50 \
    --save-best-vector ./outputs/optimized \
    --verbose

# PRISM optimization with custom parameters
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods PRISM \
    --prism-num-directions 3 \
    --prism-optimization-steps 100 \
    --prism-retain-weight 0.1 \
    --limit 50 \
    --save-best-vector ./outputs/prism_optimized \
    --verbose

# PULSE optimization with conditional steering
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods PULSE \
    --pulse-sensor-layer 15 \
    --pulse-steering-layers "12,13,14,15,16,17,18" \
    --pulse-per-layer-scaling \
    --pulse-use-entropy-scaling \
    --limit 50 \
    --save-best-vector ./outputs/pulse_optimized \
    --verbose

# TITAN optimization (full joint optimization)
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods TITAN \
    --titan-num-directions 5 \
    --titan-optimization-steps 200 \
    --titan-behavior-weight 1.0 \
    --titan-retain-weight 0.2 \
    --limit 50 \
    --save-best-vector ./outputs/titan_optimized \
    --verbose
```

### Compare All Methods
```bash
# Compare all methods on the same task
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods CAA PRISM PULSE TITAN \
    --limit 50 \
    --verbose

# Compare methods across multiple tasks
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa mmlu arc_easy \
    --methods CAA PRISM TITAN \
    --limit 50 \
    --verbose
```

### Optimize Layer for Specific Method
```bash
# Find best layer for PRISM
python -m wisent.core.main optimize-steering optimize-layer \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa \
    --method PRISM \
    --prism-num-directions 3 \
    --layer-range "8-16" \
    --strength 1.0 \
    --limit 50 \
    --verbose

# Find best layer for TITAN
python -m wisent.core.main optimize-steering optimize-layer \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa \
    --method TITAN \
    --titan-num-directions 5 \
    --layer-range "10-18" \
    --limit 50 \
    --verbose
```

### Optimize Strength for Specific Method
```bash
# Optimize PULSE strength
python -m wisent.core.main optimize-steering optimize-strength \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa \
    --method PULSE \
    --pulse-sensor-layer 15 \
    --layer 14 \
    --strength-range 0.5 3.0 \
    --strength-steps 10 \
    --limit 50 \
    --verbose
```

---

## Weight Modification

Permanently bake steering into model weights (no runtime overhead).

### CAA/PRISM Weight Modification
```bash
python -m wisent.core.main modify-weights \
    meta-llama/Llama-3.2-1B-Instruct \
    --vectors ./outputs/vectors \
    --output ./outputs/modified_model \
    --strength 1.0 \
    --norm-preserve \
    --verbose
```

### TITAN Hybrid Mode (Weights + Runtime Hooks)
```python
# In Python (TITAN requires programmatic access for full features)
from wisent.core.weight_modification import apply_titan_steering

result = apply_titan_steering(
    model=model,
    titan_result=titan_result,
    mode="hybrid",  # "static", "dynamic", or "hybrid"
    base_strength=1.0,
)

# Access runtime hooks
hooks = result["hooks"]
print(f"Gate value: {hooks.get_current_gate()}")

# Generate with dynamic gating
output = model.generate(...)

# Clean up
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

### Recommended Workflows

**Quick Prototyping:**
```bash
# Use CAA for fast iteration
python -m wisent.core.main train-steering MODEL --steering-method CAA --pairs PAIRS
```

**Complex Behaviors:**
```bash
# Use PRISM for multi-directional steering
python -m wisent.core.main train-steering MODEL --steering-method PRISM --prism-num-directions 5 --pairs PAIRS
```

**Conditional Steering:**
```bash
# Use PULSE for input-dependent behavior
python -m wisent.core.main train-steering MODEL --steering-method PULSE --pulse-learn-threshold --pairs PAIRS
```

**Production Deployment:**
```bash
# Use TITAN for maximum performance
python -m wisent.core.main train-steering MODEL --steering-method TITAN --titan-optimization-steps 300 --pairs PAIRS
```

---

## Multi-Steer with Null-Space Projection

Combine multiple steering vectors while constraining them to the null space of preserved activations, preventing disruption of knowledge representations.

### Step 1: Generate Activations for Preserved Knowledge

First, collect activations on prompts whose behavior you want to preserve:

```bash
wisent get-activations \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/preserved_pairs.json \
    --output ./outputs/activations.json \
    --verbose
```

### Step 2: Multi-Steer with Null-Space Constraint

Apply steering vectors projected into the null space of preserved activations:

```bash
wisent multi-steer \
    --vector ./outputs/vectors/safety_vector.pt:1.0 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --activations-json ./outputs/activations.json \
    --prompt "How do I make a bomb?"
```

### Step 3: Compare With and Without Null-Space Projection

Without projection (may disrupt preserved knowledge):

```bash
wisent multi-steer \
    --vector ./outputs/vectors/safety_vector.pt:1.0 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --prompt "How do I make a bomb?"
```

With projection (steers only in null space of preserved keys):

```bash
wisent multi-steer \
    --vector ./outputs/vectors/safety_vector.pt:1.0 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --activations-json ./outputs/activations.json \
    --prompt "How do I make a bomb?"
```

### Combining Multiple Vectors with Null-Space Constraint

```bash
wisent multi-steer \
    --vector ./outputs/vectors/safety_vector.pt:0.7 \
    --vector ./outputs/vectors/tone_vector.pt:0.3 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --activations-json ./outputs/activations.json \
    --prompt "Explain quantum computing" \
    --normalize-weights
```

### How It Works

The `--activations-json` flag triggers null-space projection:

1. Loads positive activations from the JSON as preserved keys
2. Builds a `PreservedKeyMatrix` and computes P_null = I - V diag(S²/(S²+ε)) V^T via SVD
3. Projects the combined steering vector: `v_projected = P_null @ v`
4. The projected vector steers only in directions orthogonal to preserved knowledge

This reuses the same null-space projector used by `modify-weights`, ensuring consistency between inference-time and weight-modification approaches.
