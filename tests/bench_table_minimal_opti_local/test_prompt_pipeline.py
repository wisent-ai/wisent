"""
Minimal test of the prompt strategy pipeline.

Tests:
1. Extract question, correct, incorrect using BoolQExtractor
2. Put into PromptFormatter.format()
3. Extract from PromptPair and convert to ContrastivePair
4. Collect activations

For ONE question only.
"""

from __future__ import annotations

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from lm_eval import tasks

from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.core.atoms import ActivationAggregationStrategy
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
from wisent.core.prompts.core.prompt_formater import PromptFormatter


def main():
    print("=" * 80)
    print("MINIMAL PROMPT PIPELINE TEST")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Extract ONE question using BoolQExtractor
    # ========================================================================
    print("\n[STEP 1] Extracting ONE question from BoolQ...")

    task_dict = tasks.get_task_dict(["sst2"])
    boolq_task = task_dict["sst2"]
    extractor = get_extractor("sst2")

    # Extract 1 pair
    pairs = extractor.extract_contrastive_pairs(
        boolq_task,
        limit=1,
        preferred_doc="validation"
    )

    if not pairs:
        print("ERROR: No pairs extracted!")
        return

    original_pair = pairs[0]
    print(f"✓ Extracted 1 pair")
    print(f"  Original prompt: {original_pair.prompt}...")
    print(f"  Correct answer: {original_pair.positive_response.model_response}")
    print(f"  Incorrect answer: {original_pair.negative_response.model_response}")

    # Extract raw data from ContrastivePair
    question = original_pair.prompt
    correct_answer = original_pair.positive_response.model_response
    incorrect_answer = original_pair.negative_response.model_response

    # ========================================================================
    # STEP 2: Put into PromptFormatter.format()
    # ========================================================================
    print("\n[STEP 2] Applying different prompt strategies...")

    formatter = PromptFormatter()

    #strategies = ["multiple_choice", "direct_completion", "role_playing", "instruction_following"]
    strategies = ["role_playing"]

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")

        # Create PromptPair
        prompt_pair = formatter.format(
            strategy=strategy,
            question=question,
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer
        )

        print(f"    PromptPair.positive:")
        print(f"      {prompt_pair.positive[0]['content']}...")
        print(f"      {prompt_pair.positive[1]['content']}")
        print(f"    PromptPair.negative:")
        print(f"      {prompt_pair.negative[0]['content']}...")
        print(f"      {prompt_pair.negative[1]['content']}")

    # ========================================================================
    # STEP 3: Convert PromptPair → ContrastivePair for ONE strategy
    # ========================================================================
    print("\n[STEP 3] Converting PromptPair → ContrastivePair (using 'multiple_choice')...")

    test_strategy = strategies[0]
    prompt_pair = formatter.format(
        strategy=test_strategy,
        question=question,
        correct_answer=correct_answer,
        incorrect_answer=incorrect_answer
    )

    # Extract from PromptPair
    print(f"\n  Extracted PromptPair messages:")
    print(f"    Positive:")
    print(f"      {prompt_pair.positive[0]['content'][:100]}...")
    print(f"      {prompt_pair.positive[1]['content']}")
    print(f"    Negative:")
    print(f"      {prompt_pair.negative[0]['content'][:100]}...")
    print(f"      {prompt_pair.negative[1]['content']}")

    # Extract content without labels (chat template handles roles)
    new_prompt = prompt_pair.positive[0]['content']
    new_positive_resp = prompt_pair.positive[1]['content']
    new_negative_resp = prompt_pair.negative[1]['content']

    # Create new ContrastivePair
    new_pair = ContrastivePair(
        prompt=new_prompt,
        positive_response=PositiveResponse(model_response=new_positive_resp),
        negative_response=NegativeResponse(model_response=new_negative_resp)
    )

    print(f"\n✓ Created ContrastivePair with '{test_strategy}' strategy")
    print(f"  Prompt: {new_pair.prompt[:100]}...")
    print(f"  Positive: {new_pair.positive_response.model_response}")
    print(f"  Negative: {new_pair.negative_response.model_response}")

    print("\n  Full new_pair before collecting activations:")
    print(f"    new_pair.prompt: {new_pair.prompt}")
    print(f"    new_pair.positive_response.model_response: {new_pair.positive_response.model_response}")
    print(f"    new_pair.negative_response.model_response: {new_pair.negative_response.model_response}")

    # ========================================================================
    # STEP 4: Collect activations
    # ========================================================================
    print("\n[STEP 4] Collecting activations...")

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"  Loading model: {model_name}")
    model = WisentModel(model_name=model_name)
    print(f"  Model loaded. Layers: {model.num_layers}")

    collector = ActivationCollector(model=model, store_device="cpu", dtype=torch.float32)

    # Collect for layer 15 with last_token aggregation
    test_layer = 15
    print(f"  Collecting activations for layer {test_layer}...")

    updated_pair = collector.collect_for_pair(
        pair=new_pair,
        layers=[str(test_layer)],
        aggregation=ActivationAggregationStrategy.LAST_TOKEN,
        return_full_sequence=False
    )

    pos_act = updated_pair.positive_response.layers_activations[str(test_layer)]
    neg_act = updated_pair.negative_response.layers_activations[str(test_layer)]

    print(f"✓ Activations collected!")
    print(f"  Positive activation shape: {pos_act.shape}")
    print(f"  Negative activation shape: {neg_act.shape}")
    print(f"  Positive activation (first 5): {pos_act[:5]}")
    print(f"  Negative activation (first 5): {neg_act[:5]}")

    # Cleanup
    del model
    del collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("✓ PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
