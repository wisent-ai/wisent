<p align="center">
  <img src="banner.png" alt="Wisent Banner" width="100%">
</p>

<p align="center">
  <code>pip install wisent</code>
</p>



## Overview

Wisent allows you to control your AI by identifying brain patterns corresponding to responses you don't like, like hallucinations or harmful outputs. We use contrastive pairs of representations to detect when a model might be generating harmful content or hallucinating. Learn more at [wisent.ai/documentation](https://www.wisent.ai/documentation).


## Null-Space Constrained Steering

Steer model behavior while preserving knowledge. The `--activations-json` flag projects steering vectors into the null space of preserved activations so they cannot disrupt representations you want to keep.

### 1. Collect activations on knowledge you want to preserve

```bash
wisent get-activations \
    meta-llama/Llama-3.2-1B-Instruct \
    --pairs ./outputs/preserved_pairs.json \
    --output ./outputs/activations.json
```

### 2. Steer with null-space constraint

```bash
wisent multi-steer \
    --vector ./outputs/vectors/safety_vector.pt:1.0 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --activations-json ./outputs/activations.json \
    --prompt "How do I make a bomb?"
```

The projector computes P_null = I - V diag(S^2/(S^2+eps)) V^T via SVD on the preserved activations, then applies `v_projected = P_null @ v` before steering. This reuses the same null-space infrastructure as `modify-weights`, ensuring consistency between inference-time and weight-modification approaches.

Without `--activations-json`, steering works as before (unconstrained additive hook).


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
