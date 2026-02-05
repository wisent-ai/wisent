"""
Test which metrics predict steering effectiveness for each method.

For each layer:
1. Detect geometry structure (LINEAR, CONE, MANIFOLD, etc.)
2. Train CAA, Probe, PRISM methods
3. Test steering effectiveness on held-out samples
4. Correlate metrics with success

Goal: Find which metric predicts success for which method.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)
import torch.nn.functional as F


def train_caa(pos_tensor, neg_tensor):
    """CAA: mean(pos) - mean(neg), normalized"""
    direction = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    return F.normalize(direction, dim=0)


def train_probe(pos_tensor, neg_tensor):
    """Linear probe direction"""
    X = torch.cat([pos_tensor, neg_tensor], dim=0).numpy()
    y = np.array([1]*len(pos_tensor) + [0]*len(neg_tensor))
    probe = LogisticRegression( random_state=42)
    probe.fit(X, y)
    direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
    return F.normalize(direction, dim=0), probe.score(X, y)


def train_prism(pos_tensor, neg_tensor, num_directions=3):
    """PRISM: multiple directions via gradient optimization"""
    caa_dir = F.normalize(pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0), dim=0)
    directions = torch.randn(num_directions, pos_tensor.shape[1])
    directions[0] = caa_dir
    for i in range(1, num_directions):
        noise = torch.randn(pos_tensor.shape[1]) * 0.3
        directions[i] = F.normalize(caa_dir + noise, dim=0)
    
    directions = F.normalize(directions, dim=1)
    directions.requires_grad_(True)
    optimizer = torch.optim.Adam([directions], lr=0.01)
    
    for step in range(100):
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


def main():
    print("="*80)
    print("GEOMETRY → STEERING TEST")
    print("Per layer: detect geometry → train recommended method → test steering")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="mps", trust_remote_code=True
    )
    model.eval()
    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"Model: {num_layers} layers, {hidden_dim} hidden dim")

    # Load data
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    all_samples = list(ds)
    train_samples = all_samples[:500]
    test_samples = all_samples[500:600]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    def get_activation(text, layer):
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return out.hidden_states[layer][0, -1, :].cpu().float()

    def collect_activations(samples, layer):
        pos_acts, neg_acts = [], []
        for sample in samples:
            correct = sample['best_answer']
            incorrect = sample['incorrect_answers'][0] if sample['incorrect_answers'] else None
            if not incorrect:
                continue
            prompt = f"Q: {sample['question']}\nA:"
            pos_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": correct}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            neg_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": incorrect}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            pos_acts.append(get_activation(pos_text, layer))
            neg_acts.append(get_activation(neg_text, layer))
        return torch.stack(pos_acts), torch.stack(neg_acts)

    def generate_with_steering(prompt, direction, strength, layer, max_tokens=50):
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

    def test_steering_effectiveness(direction, layer, test_prompts):
        """Test if steering changes outputs in expected direction"""
        changes = 0
        for prompt in test_prompts:
            base = generate_with_steering(prompt, direction, 0, layer)
            steered = generate_with_steering(prompt, direction, 15, layer)
            if base != steered:
                changes += 1
        return changes / len(test_prompts)

    # Test prompts
    test_prompts = [
        "Can you get sick from being cold?",
        "Do we only use 10% of our brain?", 
        "What happens if you swallow gum?",
        "Is it true that lightning never strikes the same place twice?",
        "Can you see the Great Wall of China from space?",
    ]

    # ============================================================
    # MAIN TEST: Per-layer geometry → method → steering
    # ============================================================
    print("\n" + "="*80)
    print("TESTING LAYERS")
    print("="*80)

    results = []
    layers_to_test = [5, 10, 15, 20, 25, 30]

    for layer in layers_to_test:
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")
        
        # 1. Collect activations
        print("Collecting activations...")
        train_pos, train_neg = collect_activations(train_samples[:200], layer)
        test_pos, test_neg = collect_activations(test_samples[:50], layer)
        print(f"  Train: {len(train_pos)} pairs, Test: {len(test_pos)} pairs")
        
        # 2. Detect geometry
        print("Detecting geometry...")
        geo_config = GeometryAnalysisConfig(num_components=5, optimization_steps=50)
        geo_result = detect_geometry_structure(train_pos, train_neg, geo_config)
        
        best_struct = geo_result.best_structure.value
        best_score = geo_result.best_score
        recommendation = geo_result.recommendation.split(" - ")[0]
        
        print(f"  Structure: {best_struct} (score={best_score:.3f})")
        print(f"  Recommendation: {recommendation}")
        
        linear_score = geo_result.all_scores.get("linear", None)
        cone_score = geo_result.all_scores.get("cone", None)
        manifold_score = geo_result.all_scores.get("manifold", None)
        
        print(f"  Scores: linear={linear_score.score if linear_score else 0:.3f}, "
              f"cone={cone_score.score if cone_score else 0:.3f}, "
              f"manifold={manifold_score.score if manifold_score else 0:.3f}")
        
        # 3. Train methods
        print("Training methods...")
        caa_dir = train_caa(train_pos, train_neg)
        probe_dir, probe_acc = train_probe(train_pos, train_neg)
        prism_dir = train_prism(train_pos, train_neg, num_directions=3)
        
        caa_probe_align = float(F.cosine_similarity(caa_dir.unsqueeze(0), probe_dir.unsqueeze(0)))
        print(f"  CAA-Probe alignment: {caa_probe_align:.3f}")
        print(f"  Probe accuracy: {probe_acc:.3f}")
        
        # 4. Test on held-out
        print("Testing on held-out...")
        X_test = torch.cat([test_pos, test_neg], dim=0).numpy()
        y_test = np.array([1]*len(test_pos) + [0]*len(test_neg))
        
        probe_full = LogisticRegression( random_state=42)
        probe_full.fit(torch.cat([train_pos, train_neg], dim=0).numpy(), 
                       np.array([1]*len(train_pos) + [0]*len(train_neg)))
        test_acc = probe_full.score(X_test, y_test)
        print(f"  Test probe accuracy: {test_acc:.3f}")
        
        test_pos_proj_caa = (test_pos @ caa_dir).numpy()
        test_neg_proj_caa = (test_neg @ caa_dir).numpy()
        caa_correct = sum(1 for p, n in zip(test_pos_proj_caa, test_neg_proj_caa) if p > n) / len(test_pos)
        
        test_pos_proj_probe = (test_pos @ probe_dir).numpy()
        test_neg_proj_probe = (test_neg @ probe_dir).numpy()
        probe_correct = sum(1 for p, n in zip(test_pos_proj_probe, test_neg_proj_probe) if p > n) / len(test_pos)
        
        test_pos_proj_prism = (test_pos @ prism_dir).numpy()
        test_neg_proj_prism = (test_neg @ prism_dir).numpy()
        prism_correct = sum(1 for p, n in zip(test_pos_proj_prism, test_neg_proj_prism) if p > n) / len(test_pos)
        
        print(f"  Projection correct: CAA={caa_correct:.2%}, Probe={probe_correct:.2%}, PRISM={prism_correct:.2%}")
        
        # 5. Test actual steering
        print("Testing steering...")
        caa_effect = test_steering_effectiveness(caa_dir, layer, test_prompts)
        probe_effect = test_steering_effectiveness(probe_dir, layer, test_prompts)
        prism_effect = test_steering_effectiveness(prism_dir, layer, test_prompts)
        
        print(f"  Steering effect: CAA={caa_effect:.2%}, Probe={probe_effect:.2%}, PRISM={prism_effect:.2%}")
        
        # 6. Store results
        results.append({
            'layer': layer,
            'geometry': best_struct,
            'geo_score': best_score,
            'linear_score': linear_score.score if linear_score else 0,
            'cone_score': cone_score.score if cone_score else 0,
            'manifold_score': manifold_score.score if manifold_score else 0,
            'recommendation': recommendation,
            'caa_probe_align': caa_probe_align,
            'probe_train_acc': probe_acc,
            'probe_test_acc': test_acc,
            'caa_proj_correct': caa_correct,
            'probe_proj_correct': probe_correct,
            'prism_proj_correct': prism_correct,
            'caa_steering_effect': caa_effect,
            'probe_steering_effect': probe_effect,
            'prism_steering_effect': prism_effect,
        })

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nPer-layer results:")
    print(f"{'Layer':>5} {'Geometry':>10} {'Score':>6} {'Lin':>5} {'Cone':>5} {'Align':>6} {'CAA%':>6} {'Probe%':>7} {'PRISM%':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['layer']:>5} {r['geometry']:>10} {r['geo_score']:>6.2f} {r['linear_score']:>5.2f} "
              f"{r['cone_score']:>5.2f} {r['caa_probe_align']:>6.2f} {r['caa_steering_effect']:>6.0%} "
              f"{r['probe_steering_effect']:>7.0%} {r['prism_steering_effect']:>7.0%}")

    print("\nCorrelation analysis:")
    
    caa_effects = [r['caa_steering_effect'] for r in results]
    probe_effects = [r['probe_steering_effect'] for r in results]
    prism_effects = [r['prism_steering_effect'] for r in results]
    alignments = [r['caa_probe_align'] for r in results]
    linear_scores = [r['linear_score'] for r in results]
    cone_scores = [r['cone_score'] for r in results]
    test_accs = [r['probe_test_acc'] for r in results]

    if len(results) > 2:
        print(f"\nCAA steering vs alignment: {np.corrcoef(caa_effects, alignments)[0,1]:.3f}")
        print(f"Probe steering vs alignment: {np.corrcoef(probe_effects, alignments)[0,1]:.3f}")
        print(f"PRISM steering vs cone_score: {np.corrcoef(prism_effects, cone_scores)[0,1]:.3f}")
        print(f"CAA steering vs linear_score: {np.corrcoef(caa_effects, linear_scores)[0,1]:.3f}")
        print(f"All steering vs test_acc: {np.corrcoef([max(c,p,pr) for c,p,pr in zip(caa_effects, probe_effects, prism_effects)], test_accs)[0,1]:.3f}")

    best_caa_layer = max(results, key=lambda r: r['caa_steering_effect'])
    best_probe_layer = max(results, key=lambda r: r['probe_steering_effect'])
    best_prism_layer = max(results, key=lambda r: r['prism_steering_effect'])

    print(f"\nBest layers:")
    print(f"  CAA: layer {best_caa_layer['layer']} ({best_caa_layer['caa_steering_effect']:.0%} effect, align={best_caa_layer['caa_probe_align']:.2f})")
    print(f"  Probe: layer {best_probe_layer['layer']} ({best_probe_layer['probe_steering_effect']:.0%} effect, test_acc={best_probe_layer['probe_test_acc']:.2f})")
    print(f"  PRISM: layer {best_prism_layer['layer']} ({best_prism_layer['prism_steering_effect']:.0%} effect, cone={best_prism_layer['cone_score']:.2f})")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
