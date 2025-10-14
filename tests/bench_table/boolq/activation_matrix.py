"""

"""

from __future__ import annotations

import torch
from lm_eval import tasks
from typing import Dict, List
import pickle
from pathlib import Path
import time
import gc

from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.activations.core.activations_collector import ActivationCollector
from wisent_guard.core.activations.core.atoms import ActivationAggregationStrategy
from wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor


# WE HAVE TO LOAD TRAIN_HF for train and valid
def load_pairs(limit: int, preferred_doc: str = "training"):
    """Load contrastive pairs from BoolQ benchmark.

    Args:
        limit: Number of pairs to load
        preferred_doc: Preferred document source ("validation", "test", "training", "fewshot")
                      Default is "training" for training data.
    """
    print(f"Loading BoolQ task...")

    task_dict = tasks.get_task_dict(["boolq"])
    boolq_task = task_dict["boolq"]

    extractor = get_extractor("boolq")

    print(f"Extracting {limit} contrastive pairs from {preferred_doc} docs...")
    pairs = extractor.extract_contrastive_pairs(boolq_task, limit=limit, preferred_doc=preferred_doc)

    print(f"Successfully extracted {len(pairs)} pairs from {preferred_doc} docs")
    return pairs

def create_activations_matrix(model_name, aggregation_methods, num_questions, output_path, preferred_doc):
    """
    Create an activations matrix where:
    - Rows: aggregation methods
    - Columns: layer numbers

    Args:
        model_name: HuggingFace model name or path
        aggregation_methods: List of aggregation methods
        num_questions: Number of contrastive pairs to extract (k)
        output_path: Optional path to save the matrix
        preferred_doc: Preferred document source ("validation", "test", "training", "fewshot")

    Returns:
        Dict containing the activations matrix and metadata
    """

    print(f"Creating activations matrix for {model_name}")

    pairs = load_pairs(limit=num_questions, preferred_doc=preferred_doc)

    if len(pairs) < num_questions:
        print(f"Warning: Only extracted {len(pairs)} pairs, expected {num_questions}")
        num_questions = len(pairs)

    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name=model_name, layers={})
    print(f"Model loaded. Hidden size: {model.hidden_size}, Layers: {model.num_layers}")

    num_layers = model.num_layers

    print(f"Dimensions: {len(aggregation_methods)} aggregations × {num_layers} layers")
    print(f"Each cell contains: {num_questions * 2} activations ({num_questions} pos + {num_questions} neg)")
    print("=" * 80)

    collector = ActivationCollector(model=model, store_device="cpu", dtype=torch.float32)

    # matrix[aggregation_method][layer] = {positive: List[tensor], negative: List[tensor]}
    matrix: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {}

    for agg_method in aggregation_methods:
        matrix[agg_method.value] = {}
        for layer_idx in range(1, num_layers + 1):
            matrix[agg_method.value][str(layer_idx)] = {
                "positive": [],
                "negative": []
            }

    print(f"\nExtracting activations for {len(pairs)} pairs...")
    print("Progress:")

    for pair_idx, pair in enumerate(pairs):
        if (pair_idx + 1) % 10 == 0 or pair_idx == 0:
            print(f"  Processing pair {pair_idx + 1}/{len(pairs)}...")

        for agg_method in aggregation_methods:

            layer_names = [str(i) for i in range(1, num_layers + 1)]

            try:
                updated_pair = collector.collect_for_pair(
                    pair=pair,
                    layers=layer_names,
                    aggregation=agg_method,
                    return_full_sequence=False, #???
                )

                pos_activations = updated_pair.positive_response.layers_activations
                neg_activations = updated_pair.negative_response.layers_activations

                for layer_idx in range(1, num_layers + 1):
                    layer_name = str(layer_idx)

                    if layer_name in pos_activations:
                        pos_tensor = pos_activations[layer_name]
                        if pos_tensor is not None:
                            matrix[agg_method.value][layer_name]["positive"].append(pos_tensor.cpu())

                    if layer_name in neg_activations:
                        neg_tensor = neg_activations[layer_name]
                        if neg_tensor is not None:
                            matrix[agg_method.value][layer_name]["negative"].append(neg_tensor.cpu())

            except Exception as e:
                print(f"  Error processing pair {pair_idx} with {agg_method.value}: {e}")
                continue

    print("\nVerifying matrix completeness...")
    for agg_method in aggregation_methods:
        for layer_idx in range(1, num_layers + 1):
            layer_name = str(layer_idx)
            n_pos = len(matrix[agg_method.value][layer_name]["positive"])
            n_neg = len(matrix[agg_method.value][layer_name]["negative"])

            if n_pos != num_questions or n_neg != num_questions:
                print(f"  Warning: [{agg_method.value}][{layer_name}] has {n_pos} pos, {n_neg} neg (expected {num_questions} each)")

    summary = {
        "model_name": model_name,
        "num_layers": num_layers,
        "num_questions": num_questions,
        "num_activations_per_cell": num_questions * 2,
        "aggregation_methods": [m.value for m in aggregation_methods],
        "hidden_size": model.hidden_size,
    }

    result = {
        "matrix": matrix,
        "summary": summary,
        "pairs": pairs,  # Include the original pairs for reference
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving matrix to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print("Matrix saved successfully!")

    print("\n" + "=" * 80)
    print("ACTIVATIONS MATRIX SUMMARY")
    print("=" * 80)
    print(f"Model: {summary['model_name']}")
    print(f"Hidden size: {summary['hidden_size']}")
    print(f"Layers: {summary['num_layers']}")
    print(f"Aggregation methods: {len(summary['aggregation_methods'])}")
    print(f"Questions per method: {summary['num_questions']}")
    print(f"Total activations per cell: {summary['num_activations_per_cell']}")
    print(f"Matrix shape: {len(aggregation_methods)} × {num_layers}")
    print("=" * 80)

    # Clean up model to free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        time.sleep(10)  # Wait for GPU memory cleanup to complete

    return result

def inspect_matrix(matrix_data: Dict):
    """
    Inspect the activations matrix and print statistics.

    Args:
        matrix_data: Dictionary returned by create_activations_matrix
    """
    matrix = matrix_data["matrix"]
    summary = matrix_data["summary"]

    print("\nMATRIX INSPECTION")
    print("=" * 80)

    # Sample a few cells
    agg_methods = summary["aggregation_methods"]
    layers_to_sample = [1, summary["num_layers"] // 2, summary["num_layers"]]

    for agg_method in agg_methods[:2]:  # Show first 2 aggregation methods
        print(f"\nAggregation method: {agg_method}")
        for layer_idx in layers_to_sample:
            layer_name = str(layer_idx)
            pos = matrix[agg_method][layer_name]["positive"]
            neg = matrix[agg_method][layer_name]["negative"]

            if pos and neg:
                print(f"  Layer {layer_name}:")
                print(f"    Positive: {len(pos)} tensors, shape {pos[0].shape}")
                print(f"    Negative: {len(neg)} tensors, shape {neg[0].shape}")

if __name__ == "__main__":
    
    aggregation_methods = [
        ActivationAggregationStrategy.CONTINUATION_TOKEN,
        ActivationAggregationStrategy.LAST_TOKEN,
        ActivationAggregationStrategy.FIRST_TOKEN,
        ActivationAggregationStrategy.MEAN_POOLING,
        ActivationAggregationStrategy.CHOICE_TOKEN,
        ActivationAggregationStrategy.MAX_POOLING,
    ]

    result = create_activations_matrix(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        aggregation_methods=aggregation_methods,
        num_questions=50,
        output_path="tests/bench_table/boolq/activations_matrix.pkl",
        preferred_doc="training"
    )

    # Inspect the matrix
    inspect_matrix(result)

    print("\nDone! The matrix has been created and saved.")
    print("You can load it later with:")
    print("  import pickle")
    print("  with open('tests/bench_table/activations_matrix.pkl', 'rb') as f:")
    print("      matrix_data = pickle.load(f)")