"""
Helper functions for geometry_steering_test.py.

Contains steering method training functions (CAA, probe, TECZA)
and steering effectiveness test utilities.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from wisent.core.constants import (
    DEFAULT_RANDOM_SEED, GEOMETRY_OPTIMIZATION_LR,
    TECZA_NUM_DIRECTIONS, DIAGNOSIS_OPTIMIZATION_STEPS,
)
from wisent.core.models.inference_config import get_generate_kwargs


def train_caa(pos_tensor, neg_tensor):
    """CAA: mean(pos) - mean(neg), normalized"""
    direction = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    return F.normalize(direction, dim=0)


def train_probe(pos_tensor, neg_tensor):
    """Linear probe direction"""
    X = torch.cat([pos_tensor, neg_tensor], dim=0).numpy()
    y = np.array([1]*len(pos_tensor) + [0]*len(neg_tensor))
    probe = LogisticRegression(random_state=DEFAULT_RANDOM_SEED)
    probe.fit(X, y)
    direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
    return F.normalize(direction, dim=0), probe.score(X, y)


def train_tecza(pos_tensor, neg_tensor, num_directions=TECZA_NUM_DIRECTIONS):
    """TECZA: multiple directions via gradient optimization"""
    caa_dir = F.normalize(pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0), dim=0)
    directions = torch.randn(num_directions, pos_tensor.shape[1])
    directions[0] = caa_dir
    for i in range(1, num_directions):
        noise = torch.randn(pos_tensor.shape[1]) * 0.3
        directions[i] = F.normalize(caa_dir + noise, dim=0)

    directions = F.normalize(directions, dim=1)
    directions.requires_grad_(True)
    optimizer = torch.optim.Adam([directions], lr=GEOMETRY_OPTIMIZATION_LR)

    for step in range(DIAGNOSIS_OPTIMIZATION_STEPS):
        optimizer.zero_grad()
        dirs_norm = F.normalize(directions, dim=1)

        pos_proj = pos_tensor @ dirs_norm.T
        neg_proj = neg_tensor @ dirs_norm.T
        separation_loss = -((pos_proj.mean(dim=0) - neg_proj.mean(dim=0)).abs().mean())

        cos_sim = dirs_norm @ dirs_norm.T
        off_diag = cos_sim * (1 - torch.eye(num_directions))
        cone_loss = F.relu(-off_diag).sum() + F.relu(off_diag - 0.95).sum()

        loss = separation_loss + 0.5 * cone_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            directions.data = F.normalize(directions.data, dim=1)

    final_dirs = F.normalize(directions.detach(), dim=1)
    return final_dirs.mean(dim=0)


def generate_with_steering(model, tokenizer, prompt, direction, strength, layer, max_tokens=None):
    """Generate text with a steering hook applied at the given layer."""
    if max_tokens is None:
        max_tokens = get_generate_kwargs()["max_new_tokens"]
    direction_tensor = direction.to(model.device).half()

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            return output + strength * direction_tensor
        elif isinstance(output, tuple):
            return (output[0] + strength * direction_tensor,) + output[1:]
        return output

    handle = model.model.layers[layer].register_forward_hook(hook)
    messages = [{"role": "user", "content": prompt + " /no_think"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def test_steering_effectiveness(model, tokenizer, direction, layer, test_prompts):
    """Test if steering changes outputs in expected direction"""
    changes = 0
    for prompt in test_prompts:
        base = generate_with_steering(model, tokenizer, prompt, direction, 0, layer)
        steered = generate_with_steering(model, tokenizer, prompt, direction, 15, layer)
        if base != steered:
            changes += 1
    return changes / len(test_prompts)
