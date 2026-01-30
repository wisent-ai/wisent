"""Behavioral label collection for steering direction discovery."""

import numpy as np


def extract_response(raw_response: str) -> str:
    """Extract assistant response from raw generation."""
    if "assistant\n\n" in raw_response:
        return raw_response.split("assistant\n\n", 1)[-1].strip()
    elif "assistant\n" in raw_response:
        return raw_response.split("assistant\n", 1)[-1].strip()
    return raw_response


def collect_behavioral_labels(adapter, test_ids, pair_texts, evaluator, layer_name, max_new_tokens=100):
    """
    Phase 1: Generate base responses, evaluate, collect activations and behavioral labels.
    Returns (activations, labels) where labels are 1=truthful, 0=untruthful.
    """
    activations = []
    labels = []

    for i, pair_key in enumerate(test_ids):
        pair_data = pair_texts[pair_key]
        prompt = pair_data.get("prompt", "")
        pos_ref_text = pair_data.get("positive", "")
        neg_ref_text = pair_data.get("negative", "")
        correct_answers = [pos_ref_text] if pos_ref_text else []
        incorrect_answers = [neg_ref_text] if neg_ref_text else []

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = adapter.apply_chat_template(messages, add_generation_prompt=True)

        # Extract activation
        layer_acts = adapter.extract_activations(formatted_prompt, layers=[layer_name])
        act = layer_acts.get(layer_name)
        if act is not None:
            activations.append(act[0, -1, :].cpu().numpy())

        # Generate and evaluate
        base_response = extract_response(adapter._generate_unsteered(
            formatted_prompt, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True
        ))
        result = evaluator.evaluate(base_response, pos_ref_text,
            correct_answers=correct_answers, incorrect_answers=incorrect_answers)
        labels.append(1 if result.ground_truth == "TRUTHFUL" else 0)

        if i % 20 == 0:
            print(f"    Collecting behavioral labels: {i+1}/{len(test_ids)}...")

    return np.array(activations), np.array(labels)


def collect_behavioral_labels_all_layers(adapter, test_ids, pair_texts, evaluator, layers, max_new_tokens=50):
    """
    Collect behavioral labels and activations for all layers at once.
    Returns (activations_by_layer, labels) where activations_by_layer maps layer_num -> numpy array.
    """
    activations_by_layer = {layer: [] for layer in layers}
    labels = []
    layer_names = [f"layer.{l}" for l in layers]

    for i, pair_key in enumerate(test_ids):
        pair_data = pair_texts[pair_key]
        prompt = pair_data.get("prompt", "")
        pos_ref_text = pair_data.get("positive", "")
        neg_ref_text = pair_data.get("negative", "")
        correct_answers = [pos_ref_text] if pos_ref_text else []
        incorrect_answers = [neg_ref_text] if neg_ref_text else []

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = adapter.apply_chat_template(messages, add_generation_prompt=True)

        # Extract activations for all layers at once
        layer_acts = adapter.extract_activations(formatted_prompt, layers=layer_names)
        for layer in layers:
            act = layer_acts.get(f"layer.{layer}")
            if act is not None:
                activations_by_layer[layer].append(act[0, -1, :].cpu().numpy())

        # Generate and evaluate
        base_response = extract_response(adapter._generate_unsteered(
            formatted_prompt, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True
        ))
        result = evaluator.evaluate(base_response, pos_ref_text,
            correct_answers=correct_answers, incorrect_answers=incorrect_answers)
        labels.append(1 if result.ground_truth == "TRUTHFUL" else 0)

        if i % 20 == 0:
            print(f"    Collecting behavioral labels (all layers): {i+1}/{len(test_ids)}...")

    # Convert to numpy arrays
    for layer in layers:
        if activations_by_layer[layer]:
            activations_by_layer[layer] = np.array(activations_by_layer[layer])
    return activations_by_layer, np.array(labels)
