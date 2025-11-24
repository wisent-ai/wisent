

# Heretic-Inspired Features for Wisent

This module implements features from the [Heretic](https://github.com/p-e-w/heretic) abliteration tool, adapted for Wisent's activation steering paradigm.

## Overview

Heretic removes censorship from LLMs using **directional ablation** (permanent weight modification). Wisent uses **activation steering** (temporary runtime modification). While the underlying techniques differ, both leverage the same insight: **specific behaviors correspond to specific directions in activation space**.

We've adapted Heretic's sophisticated optimization and analysis tools to work with Wisent's steering approach.

## Features

### 1. **RefusalDetector** - Measure Steering Effectiveness

Detects when a model refuses to answer, using configurable marker strings.

```python
from wisent.core.heretic import RefusalDetector

detector = RefusalDetector()

responses = [
    "I'm sorry, I can't help with that.",
    "The answer is 42.",
    "I cannot assist with illegal activities."
]

# Check individual responses
for response in responses:
    if detector.is_refusal(response):
        print(f"Refusal: {response}")

# Count total refusals
count = detector.count_refusals(responses)  # Returns: 2
rate = detector.get_refusal_rate(responses)  # Returns: 0.667

# Get detailed marker matches
details = detector.get_refusal_details(responses)
# Returns: {0: ['sorry', "i can't"], 2: ['cannot', 'illegal']}
```

**Default Markers** (23 total):
- "sorry", "i can't", "i cannot", "i won't", "i will not"
- "i'm unable", "i am unable", "i'm an ai", "i am an ai", "as an ai"
- "ai assistant", "i'm designed to", "i am designed to"
- "violat", "prohibit", "illegal", "harmful", "inappropriate"
- "unethical", "ethical boundaries", etc.

**Customization**:
```python
# Add custom markers
detector.add_markers(["not available", "access denied"])

# Remove markers
detector.remove_markers(["sorry"])  # Too general
```

---

### 2. **KL Divergence** - Measure Model Preservation

Quantifies how much steering changes the model's probability distribution compared to baseline.

```python
from wisent.core.heretic import compute_kl_divergence, KLDivergenceEvaluator

# One-shot computation
baseline_logprobs = model.get_logprobs(prompts)
steered_logprobs = steered_model.get_logprobs(prompts)

kl_div = compute_kl_divergence(baseline_logprobs, steered_logprobs)
print(f"KL divergence: {kl_div:.4f}")  # Lower is better
```

**Interpretation**:
- **KL = 0**: No change (perfect preservation)
- **KL < 0.1**: Minimal change (good preservation)
- **KL < 0.5**: Moderate change (acceptable)
- **KL > 1.0**: Significant change (model capabilities may be affected)

**Evaluator Class** (for repeated evaluations):
```python
# Create evaluator with baseline model
evaluator = KLDivergenceEvaluator(
    model=baseline_model,
    prompts=harmless_prompts,  # Should be benign/safe prompts
)

# Evaluate steering at different strengths
for alpha in [0.5, 1.0, 2.0]:
    kl = evaluator.evaluate_with_steering(model, steering_vectors, alpha)
    print(f"α={alpha}: KL={kl:.4f}")
```

---

### 3. **Direction Interpolation** - Smooth Cross-Layer Steering

Interpolate steering vectors at float layer indices for smoother steering.

```python
from wisent.core.heretic import interpolate_steering_vector, get_global_steering_direction

# Steering vectors computed per-layer
steering_vectors = {
    0: torch.randn(4096),
    1: torch.randn(4096),
    # ... layers 2-31
}

# Interpolate at float index
vector_15_5 = interpolate_steering_vector(steering_vectors, 15.5)
# Returns: 0.5 * steering_vectors[15] + 0.5 * steering_vectors[16]

# Use global direction (Heretic-style)
global_vectors = get_global_steering_direction(steering_vectors, 15.5)
# All layers now use the interpolated direction from layer 15.5
```

**Use Cases**:
1. **Find optimal layer**: Try `direction_index` in [10.0, 20.0] to find best single direction
2. **Smooth transitions**: Use float indices to avoid discontinuities between layers
3. **Global steering**: Apply single coherent direction across all layers

---

### 4. **Geometry Analyzer** - Understand Your Steering Vectors

Analyzes how well-separated positive and negative activation distributions are, and how well steering vectors align with class means.

```python
from wisent.core.heretic import analyze_steering_geometry, GeometryAnalyzer

# Compute geometry metrics
metrics = analyze_steering_geometry(
    positive_activations,   # {layer: [N, H] tensor}
    negative_activations,   # {layer: [M, H] tensor}
    steering_vectors,       # {layer: [H] tensor}
    print_table=True,       # Print formatted table
)
```

**Output**:
```
========================================================================================
STEERING GEOMETRY ANALYSIS
========================================================================================
 Layer |    |Pos|    |Neg| |Steer| | S(p,n) S(p,s) S(n,s) | SepQual
----------------------------------------------------------------------------------------
     0 |   145.23   142.67   289.45 |   0.856   0.923  -0.067 |    0.990
     1 |   148.91   145.12   294.03 |   0.861   0.928  -0.067 |    0.995
   ...
    31 |   152.34   149.55   301.89 |   0.867   0.932  -0.065 |    0.997
========================================================================================

Legend:
  |Pos|, |Neg|, |Steer|: L2 norms
  S(p,n): Cosine similarity between positive and negative means
  S(p,s): Cosine similarity between positive mean and steering vector
  S(n,s): Cosine similarity between negative mean and steering vector
  SepQual: Separation quality = S(p,s) - S(n,s) (higher is better)
```

**Metrics Interpretation**:

- **Separation Quality (SepQual)**: Higher is better
  - `> 0.5`: Excellent separation
  - `0.2 - 0.5`: Good separation
  - `< 0.2`: Poor separation (steering may not work well)

- **S(p,n)**: Positive-Negative similarity
  - `< 0.7`: Well-separated classes (good)
  - `> 0.9`: Poorly separated (bad)

- **S(p,s)**: Positive-Steering similarity
  - Should be high (> 0.8) - steering points toward positive

- **S(n,s)**: Negative-Steering similarity
  - Should be negative or low - steering away from negative

**Programmatic Analysis**:
```python
analyzer = GeometryAnalyzer()
metrics = analyzer.analyze(positive_acts, negative_acts, steering_vecs)

# Get summary statistics
summary = analyzer.get_summary_statistics(metrics)
print(summary)
# {'mean_separation_quality': 0.95, 'min_separation_quality': 0.82, ...}

# Find best layers
best_layers = analyzer.identify_best_layers(metrics, top_k=5)
# [(15, 0.998), (16, 0.995), (14, 0.992), ...]

# Find problematic layers
problems = analyzer.identify_problematic_layers(metrics, threshold=0.1)
# [(2, 'low separation (0.08); weak steering (0.45)'), ...]
```

---

### 5. **Multi-Objective Optimizer** - Automatic Parameter Tuning

Uses Optuna's TPE (Tree-structured Parzen Estimator) to automatically find optimal steering parameters by balancing:
1. **Task performance** (accuracy, F1, etc.)
2. **KL divergence** (model preservation)

This creates a **Pareto frontier** of solutions - you cannot improve one objective without worsening the other.

```python
from wisent.core.heretic import MultiObjectiveOptimizer, SteeringParameters

# Define evaluation function
def evaluate(params: SteeringParameters) -> tuple[float, float]:
    """
    Apply steering with params and evaluate.

    Returns:
        (task_metric, kl_divergence)
        - task_metric: Higher is better (e.g., accuracy)
        - kl_divergence: Lower is better
    """
    # Apply steering with params.alpha, params.direction_index, etc.
    accuracy = evaluate_task_accuracy(model, params)
    kl_div = evaluate_kl_divergence(model, params)

    return accuracy, kl_div

# Create optimizer
optimizer = MultiObjectiveOptimizer(
    model=model,
    steering_vectors=steering_vectors,
    evaluate_fn=evaluate,
    direction="maximize",  # Maximize task_metric
)

# Run optimization (100 trials, ~10 minutes on RTX 3090)
result = optimizer.optimize(
    n_trials=100,
    n_startup_trials=30,  # Random exploration
)

# Print Pareto frontier
optimizer.print_pareto_frontier(result)
```

**Output**:
```
================================================================================
PARETO-OPTIMAL TRIALS
================================================================================
 Trial | Task Metric |    KL Div |    Alpha |      Scope
--------------------------------------------------------------------------------
     8 |      0.9523 |    0.0824 |     0.85 |  per_layer
    23 |      0.9614 |    0.1255 |     1.23 | global
    47 |      0.9458 |    0.0512 |     0.62 |  per_layer
    89 |      0.9701 |    0.2104 |     2.15 | global
================================================================================
```

**Select Best Trial**:
```python
# Automatically select best trial (weighted score)
best_params = optimizer.select_best_trial(
    result,
    task_weight=0.7,  # 70% weight on task, 30% on KL
)

print(best_params)
# SteeringParameters(alpha=0.85, scope=per_layer, dir_idx=None)

# Apply best parameters
apply_steering(model, steering_vectors, alpha=best_params.alpha)
```

**Optimized Parameters**:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `alpha` | [0.1, 10.0] (log scale) | Global steering strength |
| `direction_scope` | {global, per_layer} | Use single direction or per-layer |
| `direction_index` | [0.4*(L-1), 0.9*(L-1)] | Float layer index for global direction |
| `use_layer_weights` | {True, False} | Enable per-layer scaling |
| `max_weight` | [0.5, 2.0] | Peak layer weight |
| `max_weight_position` | [0.4*(L-1), L-1] | Layer of peak weight |
| `min_weight` | [0.0, max_weight] | Minimum layer weight |
| `min_weight_distance` | [1.0, 0.6*(L-1)] | Weight decay distance |

**Quick Alpha-Only Optimization**:
```python
from wisent.core.heretic.multi_objective_optimizer import quick_optimize_steering

def evaluate_alpha(alpha: float) -> tuple[float, float]:
    return task_accuracy, kl_divergence

best_alpha, accuracy, kl = quick_optimize_steering(
    model, steering_vectors, evaluate_alpha, n_trials=50
)

print(f"Best α: {best_alpha:.2f}, Accuracy: {accuracy:.2%}, KL: {kl:.4f}")
```

---

## Complete Example: Optimizing Steering for Math Task

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent.core.heretic import (
    MultiObjectiveOptimizer,
    KLDivergenceEvaluator,
    analyze_steering_geometry,
    SteeringParameters,
)

# 1. Load model and compute steering vectors
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# (Compute steering_vectors using your existing Wisent pipeline)
# steering_vectors = {...}

# 2. Create KL divergence evaluator
kl_evaluator = KLDivergenceEvaluator(
    model=model,
    prompts=[
        "What is 2+2?",
        "Solve: 5 * 7 = ?",
        # ... more harmless math prompts
    ],
)

# 3. Define evaluation function
def evaluate(params: SteeringParameters) -> tuple[float, float]:
    # Apply steering
    from wisent.core.steering.application import apply_steering
    handles = apply_steering(model, steering_vectors, params.alpha)

    # Evaluate task accuracy
    correct = 0
    for prompt, expected in test_cases:
        response = model.generate(prompt)
        if expected in response:
            correct += 1
    accuracy = correct / len(test_cases)

    # Evaluate KL divergence
    kl_div = kl_evaluator.evaluate(model)

    # Remove steering
    for handle in handles:
        handle.remove()

    return accuracy, kl_div

# 4. Run multi-objective optimization
optimizer = MultiObjectiveOptimizer(
    model=model,
    steering_vectors=steering_vectors,
    evaluate_fn=evaluate,
    direction="maximize",
)

result = optimizer.optimize(n_trials=100)
optimizer.print_pareto_frontier(result)

# 5. Select and apply best parameters
best_params = optimizer.select_best_trial(result, task_weight=0.7)
print(f"Best parameters: {best_params}")

# 6. Analyze geometry of final solution
analyze_steering_geometry(
    positive_activations,
    negative_activations,
    steering_vectors,
    print_table=True,
)
```

---

## Comparison: Heretic vs Wisent

| Feature | Heretic | Wisent (This Module) |
|---------|---------|---------------------|
| **Core Method** | Weight orthogonalization | Activation steering |
| **Permanence** | Permanent | Temporary (hooks) |
| **Use Case** | Remove safety filters | Task-specific improvements |
| **Optimization** | 9D parameter space (200 trials) | 1-8D parameter space (50-100 trials) |
| **Runtime** | 45 min (Llama-3.1-8B, RTX 3090) | 5-10 min |
| **Artifact** | Modified model (8-70GB) | Steering vectors (MB) |
| **Reversibility** | No | Yes |
| **Multi-task** | No | Yes |

**Key Adaptation**: Heretic's weight kernel becomes our per-layer scaling in activation space.

---

## Installation

Requires Optuna for multi-objective optimization:

```bash
pip install optuna
```

All other dependencies are included with Wisent.

---

## Running the Demo

```bash
python -m wisent.examples.heretic_features_demo
```

This runs 5 demonstrations showcasing all features with synthetic data.

---

## Acknowledgments

These features are adapted from [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann (pew@worldwidemann.com).

- Original paper: Arditi et al. 2024 - "Refusal in Language Models Is Mediated by a Single Direction"
- Heretic repository: https://github.com/p-e-w/heretic
- License: AGPL-3.0

Our adaptations are licensed under Wisent's license.

---

## Citation

If you use these features, please cite both Wisent and Heretic:

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```

---

## Future Work

Potential enhancements:
- Batch size auto-optimization
- Streaming evaluation for large datasets
- Distributed optimization across multiple GPUs
- Integration with Weights & Biases for tracking
- Pre-computed geometry metrics caching
- Automatic layer weight kernel optimization

---

## Support

For issues or questions:
1. Check the demo: `python -m wisent.examples.heretic_features_demo`
2. Read the comparison: `HERETIC_VS_WISENT_COMPARISON.md`
3. File an issue on GitHub

---

**Happy steering! 🎯**
