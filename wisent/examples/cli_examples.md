# Wisent CLI Examples

This document provides CLI examples for steering methods available in Wisent.
See also: [Weight Modification Examples](cli_weight_modification.md)

## Table of Contents
- [Basic Steering (CAA)](#basic-steering-caa)
- [PRISM - Multi-Directional Steering](#prism---multi-directional-steering)
- [PULSE - Conditional Layer-Adaptive Steering](#pulse---conditional-layer-adaptive-steering)
- [TITAN - Joint Optimization Steering](#titan---joint-optimization-steering)
- [Optimization Commands](#optimization-commands)

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

PRISM discovers multiple steering directions per layer using gradient optimization.

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
| `--prism-num-directions` | 3 | Number of directions per layer |
| `--prism-optimization-steps` | 100 | Gradient steps |
| `--prism-learning-rate` | 0.01 | Step size for optimization |
| `--prism-retain-weight` | 0.1 | Preserve behavior on non-target examples |
| `--prism-independence-weight` | 0.05 | Encourage diverse directions |
| `--prism-use-caa-init` | True | Initialize with CAA direction |
| `--prism-cone-constraint` | True | Keep directions in same half-space |

---

## PULSE - Conditional Layer-Adaptive Steering

PULSE applies steering conditionally based on input content, with learned per-layer scaling.

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
| `--pulse-gate-temperature` | 0.1 | Sigmoid sharpness |
| `--pulse-per-layer-scaling` | True | Learn layer-specific strengths |
| `--pulse-use-entropy-scaling` | True | Modulate by model uncertainty |
| `--pulse-max-alpha` | 2.0 | Maximum steering intensity |

---

## TITAN - Joint Optimization Steering

TITAN jointly optimizes multi-directional manifold discovery, learned gating, per-input intensity, and direction weighting.

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

The `optimize-steering` command supports all steering methods (CAA, PRISM, PULSE, TITAN).

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

# TITAN optimization (full joint optimization)
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods TITAN \
    --titan-num-directions 5 \
    --titan-optimization-steps 200 \
    --limit 50 \
    --save-best-vector ./outputs/titan_optimized \
    --verbose
```

### Compare All Methods
```bash
python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa \
    --methods CAA PRISM PULSE TITAN \
    --limit 50 \
    --verbose
```

### Optimize Layer / Strength
```bash
# Find best layer for PRISM
python -m wisent.core.main optimize-steering optimize-layer \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa --method PRISM \
    --prism-num-directions 3 --layer-range "8-16" --limit 50 --verbose

# Optimize PULSE strength
python -m wisent.core.main optimize-steering optimize-strength \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa --method PULSE \
    --pulse-sensor-layer 15 --layer 14 \
    --strength-range 0.5 3.0 --strength-steps 10 --limit 50 --verbose
```
