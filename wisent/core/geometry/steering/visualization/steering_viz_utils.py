"""Utility functions for steering visualization pipeline."""

import json
import numpy as np
import torch
from pathlib import Path
from argparse import Namespace
from typing import Tuple


def create_steering_object_from_pairs(args, tmpdir: Path) -> str:
    """Create a steering object from contrastive pairs in database."""
    from wisent.core.cli.get_activations import execute_get_activations
    from wisent.core.cli.create_steering_object import execute_create_steering_object
    from wisent.core.geometry.repscan_with_concepts import load_pair_texts_from_database

    print("  Creating steering object from contrastive pairs...")
    pairs_path = tmpdir / "pairs.json"
    pair_texts = load_pair_texts_from_database(
        task_name=args.task, limit=getattr(args, 'limit', 100),
        database_url=getattr(args, 'database_url', None)
    )
    pairs_list = [{"prompt": p.get("prompt", ""),
                   "positive_response": {"model_response": p.get("positive", "")},
                   "negative_response": {"model_response": p.get("negative", "")}}
                  for p in pair_texts.values()]
    with open(pairs_path, 'w') as f:
        json.dump({"task_name": args.task, "pairs": pairs_list}, f)

    enriched_path = tmpdir / "enriched_pairs.json"
    execute_get_activations(Namespace(
        pairs_file=str(pairs_path), model=args.model, output=str(enriched_path),
        device=getattr(args, 'device', None),
        layers=str(args.layer) if hasattr(args, 'layer') else 'all',
        extraction_strategy=getattr(args, 'extraction_strategy', 'chat_last'),
        verbose=False, timing=False, limit=None, raw=False,
    ))

    steering_path = tmpdir / "steering_object.pt"
    execute_create_steering_object(Namespace(
        enriched_pairs_file=str(enriched_path), method='caa', output=str(steering_path),
        layer=str(args.layer) if hasattr(args, 'layer') else None,
        normalize=True, verbose=False, timing=False,
    ))
    return str(steering_path)


def extract_activations_from_responses(base_data: dict, steered_data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract activations from response data JSONs."""
    base_acts, steered_acts = [], []
    for base_resp, steered_resp in zip(base_data['responses'], steered_data['responses']):
        base_act = base_resp.get('activations', {})
        steered_act = steered_resp.get('activations', {})
        if base_act and steered_act:
            base_vec = list(base_act.values())[0] if base_act else None
            steered_vec = list(steered_act.values())[0] if steered_act else None
            if base_vec and steered_vec:
                base_acts.append(torch.tensor(base_vec))
                steered_acts.append(torch.tensor(steered_vec))
    if not base_acts:
        raise ValueError("No activations found in responses.")
    return torch.stack(base_acts), torch.stack(steered_acts)


def load_reference_activations(args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load reference activations from database for classifier training."""
    from wisent.core.geometry.repscan_with_concepts import load_activations_from_database
    return load_activations_from_database(
        model_name=args.model, task_name=args.task, layer=args.layer,
        prompt_format=getattr(args, 'prompt_format', 'chat'),
        extraction_strategy=getattr(args, 'extraction_strategy', 'chat_last'),
        limit=getattr(args, 'limit', 100),
        database_url=getattr(args, 'database_url', None),
    )


def train_classifier_and_predict(pos_ref, neg_ref, base_activations, steered_activations, classifier_type='mlp'):
    """Train classifier on reference data and predict on response activations."""
    from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
    from wisent.core.classifiers.classifiers.models.mlp import MLPClassifier
    from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig

    X_train = torch.cat([pos_ref, neg_ref], dim=0).cpu().numpy()
    y_train = np.concatenate([np.ones(len(pos_ref)), np.zeros(len(neg_ref))])
    classifier = MLPClassifier(device="cpu", hidden_dim=256) if classifier_type == "mlp" else LogisticClassifier(device="cpu")
    train_report = classifier.fit(X_train, y_train, config=ClassifierTrainConfig(test_size=0.2, num_epochs=100, batch_size=32))
    base_probs = classifier.predict_proba(base_activations.cpu().numpy())
    steered_probs = classifier.predict_proba(steered_activations.cpu().numpy())
    base_probs = base_probs if isinstance(base_probs, list) else [base_probs]
    steered_probs = steered_probs if isinstance(steered_probs, list) else [steered_probs]
    return base_probs, steered_probs, train_report


def save_viz_summary(output_path: Path, args, base_evaluations, steered_evaluations,
                     base_space_probs, steered_space_probs, train_report, base_data, steered_data):
    """Save JSON summary of visualization results."""
    base_truthful = sum(1 for e in base_evaluations if e == "TRUTHFUL")
    steered_truthful = sum(1 for e in steered_evaluations if e == "TRUTHFUL")
    all_responses = [{"prompt": b.get("prompt"), "positive_reference": b.get("positive_reference"),
                      "negative_reference": b.get("negative_reference"),
                      "base_response": b.get("generated_response"), "steered_response": s.get("generated_response")}
                     for b, s in zip(base_data['responses'], steered_data['responses'])]
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({"model": args.model, "task": args.task, "layer": args.layer, "strength": args.strength,
                   "text_evaluation": {"base_truthful": base_truthful, "steered_truthful": steered_truthful, "total": len(base_evaluations)},
                   "activation_space_location": {"classifier_accuracy": train_report.final.accuracy, "classifier_auc": train_report.final.auc,
                       "base_in_truthful_region": sum(1 for p in base_space_probs if p >= 0.5),
                       "steered_in_truthful_region": sum(1 for p in steered_space_probs if p >= 0.5),
                       "total": len(base_space_probs), "base_mean_prob": float(np.mean(base_space_probs)),
                       "steered_mean_prob": float(np.mean(steered_space_probs))},
                   "responses": all_responses}, f, indent=2)
    print(f"Summary saved to: {json_path}")


def extract_base_and_steered_activations(
    wisent,
    prompts,
    steering_vectors,
    layer: int,
    steering_strength: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract activations before and after steering for a set of prompts."""
    from wisent.core.adapters.base import SteeringConfig

    adapter = wisent.adapter
    layer_name = f"layer.{layer}"

    base_acts = []
    steered_acts = []

    for prompt in prompts:
        base_layer_acts = adapter.extract_activations(prompt, layers=[layer_name])
        base_act = base_layer_acts.get(layer_name)
        if base_act is not None:
            base_acts.append(base_act[0, -1, :])

        steered_act = _extract_with_steering(
            adapter, prompt, layer_name, steering_vectors,
            SteeringConfig(strength=steering_strength)
        )
        if steered_act is not None:
            steered_acts.append(steered_act)

    if not base_acts or not steered_acts:
        raise ValueError("No activations extracted")

    return torch.stack(base_acts), torch.stack(steered_acts)


def _extract_with_steering(adapter, prompt, layer_name, steering_vectors, config):
    """Extract activations from a single forward pass with steering applied."""
    from wisent.core.modalities import TextContent

    content = TextContent(text=prompt) if isinstance(prompt, str) else prompt

    inputs = adapter.tokenizer(content.text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(adapter.model.device) for k, v in inputs.items()}

    activation_storage = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activation_storage['activation'] = output.detach().cpu()

    all_points = {ip.name: ip for ip in adapter.get_intervention_points()}
    if layer_name not in all_points:
        return None

    ip = all_points[layer_name]
    module = adapter._get_module_by_path(ip.module_path)
    if module is None:
        return None

    try:
        capture_handle = module.register_forward_hook(capture_hook)

        with adapter._steering_hooks(steering_vectors, config):
            with torch.no_grad():
                adapter.model(**inputs)

        capture_handle.remove()

        if 'activation' in activation_storage:
            return activation_storage['activation'][0, -1, :]
        return None

    except Exception as e:
        print(f"Error extracting steered activation: {e}")
        return None
