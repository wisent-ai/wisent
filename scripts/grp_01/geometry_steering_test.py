"""
Test which metrics predict steering effectiveness for each method.

For each layer:
1. Detect geometry structure (LINEAR, CONE, MANIFOLD, etc.)
2. Train CAA, Probe, TECZA methods
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
from wisent.core.constants import (
    TOKENIZER_MAX_LENGTH_GEOMETRY,
    DEFAULT_RANDOM_SEED,
    GEOMETRY_OPTIMIZATION_STEPS_DEFAULT,
    TECZA_NUM_DIRECTIONS,
)

from geometry_steering_test_helpers import (
    train_caa,
    train_probe,
    train_tecza,
    test_steering_effectiveness,
)


def main():
    print("="*80)
    print("GEOMETRY -> STEERING TEST")
    print("Per layer: detect geometry -> train recommended method -> test steering")
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
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LENGTH_GEOMETRY)
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

    test_prompts = [
        "Can you get sick from being cold?",
        "Do we only use 10% of our brain?",
        "What happens if you swallow gum?",
        "Is it true that lightning never strikes the same place twice?",
        "Can you see the Great Wall of China from space?",
    ]

    print("\n" + "="*80)
    print("TESTING LAYERS")
    print("="*80)

    results = []
    layers_to_test = [5, 10, 15, 20, 25, 30]

    for layer in layers_to_test:
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        print("Collecting activations...")
        train_pos, train_neg = collect_activations(train_samples[:200], layer)
        test_pos, test_neg = collect_activations(test_samples[:50], layer)
        print(f"  Train: {len(train_pos)} pairs, Test: {len(test_pos)} pairs")

        print("Detecting geometry...")
        geo_config = GeometryAnalysisConfig(num_components=5, optimization_steps=GEOMETRY_OPTIMIZATION_STEPS_DEFAULT)
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

        print("Training methods...")
        caa_dir = train_caa(train_pos, train_neg)
        probe_dir, probe_acc = train_probe(train_pos, train_neg)
        tecza_dir = train_tecza(train_pos, train_neg, num_directions=TECZA_NUM_DIRECTIONS)

        caa_probe_align = float(torch.nn.functional.cosine_similarity(caa_dir.unsqueeze(0), probe_dir.unsqueeze(0)))
        print(f"  CAA-Probe alignment: {caa_probe_align:.3f}")
        print(f"  Probe accuracy: {probe_acc:.3f}")

        print("Testing on held-out...")
        X_test = torch.cat([test_pos, test_neg], dim=0).numpy()
        y_test = np.array([1]*len(test_pos) + [0]*len(test_neg))

        probe_full = LogisticRegression(random_state=DEFAULT_RANDOM_SEED)
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

        test_pos_proj_tecza = (test_pos @ tecza_dir).numpy()
        test_neg_proj_tecza = (test_neg @ tecza_dir).numpy()
        tecza_correct = sum(1 for p, n in zip(test_pos_proj_tecza, test_neg_proj_tecza) if p > n) / len(test_pos)

        print(f"  Projection correct: CAA={caa_correct:.2%}, Probe={probe_correct:.2%}, TECZA={tecza_correct:.2%}")

        print("Testing steering...")
        caa_effect = test_steering_effectiveness(model, tokenizer, caa_dir, layer, test_prompts)
        probe_effect = test_steering_effectiveness(model, tokenizer, probe_dir, layer, test_prompts)
        tecza_effect = test_steering_effectiveness(model, tokenizer, tecza_dir, layer, test_prompts)

        print(f"  Steering effect: CAA={caa_effect:.2%}, Probe={probe_effect:.2%}, TECZA={tecza_effect:.2%}")

        results.append({
            'layer': layer, 'geometry': best_struct, 'geo_score': best_score,
            'linear_score': linear_score.score if linear_score else 0,
            'cone_score': cone_score.score if cone_score else 0,
            'manifold_score': manifold_score.score if manifold_score else 0,
            'recommendation': recommendation, 'caa_probe_align': caa_probe_align,
            'probe_train_acc': probe_acc, 'probe_test_acc': test_acc,
            'caa_proj_correct': caa_correct, 'probe_proj_correct': probe_correct,
            'tecza_proj_correct': tecza_correct,
            'caa_steering_effect': caa_effect, 'probe_steering_effect': probe_effect,
            'tecza_steering_effect': tecza_effect,
        })

    # SUMMARY
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nPer-layer results:")
    print(f"{'Layer':>5} {'Geometry':>10} {'Score':>6} {'Lin':>5} {'Cone':>5} {'Align':>6} {'CAA%':>6} {'Probe%':>7} {'TECZA%':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['layer']:>5} {r['geometry']:>10} {r['geo_score']:>6.2f} {r['linear_score']:>5.2f} "
              f"{r['cone_score']:>5.2f} {r['caa_probe_align']:>6.2f} {r['caa_steering_effect']:>6.0%} "
              f"{r['probe_steering_effect']:>7.0%} {r['tecza_steering_effect']:>7.0%}")

    if len(results) > 2:
        caa_effects = [r['caa_steering_effect'] for r in results]
        probe_effects = [r['probe_steering_effect'] for r in results]
        tecza_effects = [r['tecza_steering_effect'] for r in results]
        alignments = [r['caa_probe_align'] for r in results]
        linear_scores = [r['linear_score'] for r in results]
        cone_scores = [r['cone_score'] for r in results]
        test_accs = [r['probe_test_acc'] for r in results]

        print(f"\nCorrelation analysis:")
        print(f"CAA steering vs alignment: {np.corrcoef(caa_effects, alignments)[0,1]:.3f}")
        print(f"Probe steering vs alignment: {np.corrcoef(probe_effects, alignments)[0,1]:.3f}")
        print(f"TECZA steering vs cone_score: {np.corrcoef(tecza_effects, cone_scores)[0,1]:.3f}")
        print(f"CAA steering vs linear_score: {np.corrcoef(caa_effects, linear_scores)[0,1]:.3f}")
        print(f"All steering vs test_acc: {np.corrcoef([max(c,p,pr) for c,p,pr in zip(caa_effects, probe_effects, tecza_effects)], test_accs)[0,1]:.3f}")

    best_caa_layer = max(results, key=lambda r: r['caa_steering_effect'])
    best_probe_layer = max(results, key=lambda r: r['probe_steering_effect'])
    best_tecza_layer = max(results, key=lambda r: r['tecza_steering_effect'])

    print(f"\nBest layers:")
    print(f"  CAA: layer {best_caa_layer['layer']} ({best_caa_layer['caa_steering_effect']:.0%} effect, align={best_caa_layer['caa_probe_align']:.2f})")
    print(f"  Probe: layer {best_probe_layer['layer']} ({best_probe_layer['probe_steering_effect']:.0%} effect, test_acc={best_probe_layer['probe_test_acc']:.2f})")
    print(f"  TECZA: layer {best_tecza_layer['layer']} ({best_tecza_layer['tecza_steering_effect']:.0%} effect, cone={best_tecza_layer['cone_score']:.2f})")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
