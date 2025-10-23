"""
Activation Matrix Utilities for Minimal Optimization.

Handles loading contrastive pairs and creating activation matrices
for different aggregation methods and prompt strategies.
"""

from __future__ import annotations

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from lm_eval import tasks
from typing import Dict, List, Tuple
import pickle

from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.core.atoms import ActivationAggregationStrategy
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent.core.prompts.core.prompt_formater import PromptFormatter
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse


def load_pairs(benchmark: str, limit: int, preferred_doc: str = "training"):
    """Load contrastive pairs from benchmark.

    Args:
        benchmark: Name of benchmark
        limit: Number of pairs to load
        preferred_doc: Preferred document source ("validation", "test", "training", "fewshot")

    Returns:
        List of contrastive pairs
    """
    print(f"Loading {benchmark} task...")
    task_dict = tasks.get_task_dict([benchmark])
    benchmark_task = task_dict[benchmark]
    extractor = get_extractor(benchmark)

    print(f"Extracting {limit} contrastive pairs from {preferred_doc} docs for benchmark {benchmark}")
    pairs = extractor.extract_contrastive_pairs(benchmark_task, limit=limit, preferred_doc=preferred_doc)
    print(f"Successfully extracted {len(pairs)} pairs from {preferred_doc} docs")
    return pairs


def create_prompt_pair(question: str, correct_answer: str, incorrect_answer: str, prompt_strategy: str):
    """Create a prompt pair using the specified strategy.

    Args:
        question: The question text
        correct_answer: The correct answer
        incorrect_answer: The incorrect answer
        prompt_strategy: Prompt construction strategy name

    Returns:
        PromptPair object with positive and negative prompts
    """
    formatter = PromptFormatter()
    return formatter.format(
        strategy=prompt_strategy,
        question=question,
        correct_answer=correct_answer,
        incorrect_answer=incorrect_answer
    )


def extract_raw_data_from_pair(pair: ContrastivePair) -> Tuple[str, str, str]:
    """Extract raw question, correct answer, and incorrect answer from ContrastivePair.

    Args:
        pair: ContrastivePair with prompt and responses

    Returns:
        Tuple of (question, correct_answer, incorrect_answer)
    """
    question = pair.prompt
    correct_answer = pair.positive_response.model_response
    incorrect_answer = pair.negative_response.model_response
    return question, correct_answer, incorrect_answer


def convert_prompt_pair_to_contrastive_pair(prompt_pair) -> ContrastivePair:
    """Convert PromptPair to ContrastivePair.

    Args:
        prompt_pair: PromptPair with positive and negative chat messages

    Returns:
        ContrastivePair with extracted prompt and responses
    """
    # Extract content from PromptPair (chat template will handle role formatting)
    new_prompt = prompt_pair.positive[0]['content']
    new_positive_resp = prompt_pair.positive[1]['content']
    new_negative_resp = prompt_pair.negative[1]['content']

    return ContrastivePair(
        prompt=new_prompt,
        positive_response=PositiveResponse(model_response=new_positive_resp),
        negative_response=NegativeResponse(model_response=new_negative_resp)
    )


def extract_activations_for_config(
    model_name: str,
    benchmark: str,
    pairs: List,
    layer: int,
    aggregation_method: ActivationAggregationStrategy,
    prompt_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract activations for a specific configuration.

    Args:
        model_name: HuggingFace model name
        benchmark: Benchmark name
        pairs: List of contrastive pairs
        layer: Layer index to extract from
        aggregation_method: Aggregation strategy
        prompt_strategy: Prompt construction strategy

    Returns:
        Tuple of (positive_activations, negative_activations) as tensors
        Shape: [num_pairs, hidden_size]
    """
    print(f"\nExtracting activations:")
    print(f"  Layer: {layer}")
    print(f"  Aggregation: {aggregation_method.value}")
    print(f"  Prompt strategy: {prompt_strategy}")

    model = WisentModel(model_name=model_name)
    collector = ActivationCollector(model=model, store_device="cpu", dtype=torch.float32)

    positive_acts = []
    negative_acts = []

    for pair_idx, pair in enumerate(pairs):
        if (pair_idx + 1) % 50 == 0:
            print(f"  Processing pair {pair_idx + 1}/{len(pairs)}...")

        try:
            # Extract raw data from ContrastivePair
            question, correct_answer, incorrect_answer = extract_raw_data_from_pair(pair)

            # Create PromptPair with specified strategy
            prompt_pair = create_prompt_pair(
                question=question,
                correct_answer=correct_answer,
                incorrect_answer=incorrect_answer,
                prompt_strategy=prompt_strategy
            )

            # Convert PromptPair back to ContrastivePair
            reconstructed_pair = convert_prompt_pair_to_contrastive_pair(prompt_pair)

            # Collect activations
            updated_pair = collector.collect_for_pair(
                pair=reconstructed_pair,
                layers=[str(layer)],
                aggregation=aggregation_method,
                return_full_sequence=False,
            )

            pos_act = updated_pair.positive_response.layers_activations[str(layer)]
            neg_act = updated_pair.negative_response.layers_activations[str(layer)]

            if pos_act is not None and neg_act is not None:
                positive_acts.append(pos_act.cpu())
                negative_acts.append(neg_act.cpu())

        except Exception as e:
            print(f"  Error processing pair {pair_idx}: {e}")
            continue

    # Delete model and collector (caller will do aggressive cleanup)
    del model
    del collector

    # Stack into tensors
    positive_tensor = torch.stack(positive_acts) if positive_acts else torch.tensor([])
    negative_tensor = torch.stack(negative_acts) if negative_acts else torch.tensor([])

    print(f"  Extracted {len(positive_acts)} positive and {len(negative_acts)} negative activations")

    return positive_tensor, negative_tensor
