#!/usr/bin/env python3
"""
Test how chat template extraction strategies work visually.

Shows the difference between raw text and chat-templated text
for contrastive pairs from BoolQ using the actual wisent extractor.
"""
import torch
from transformers import AutoTokenizer

from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.boolq import BoolQExtractor
from wisent.core.activations import (
    ExtractionStrategy,
    build_extraction_texts,
)
from wisent.core.models.wisent_model import WisentModel

# Small models < 1GB
CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # instruct/chat model with chat template
BASE_MODEL = "Qwen/Qwen2.5-0.5B"            # base model without chat template

DEVICE = "cuda"
N_SAMPLES = 2


def load_boolq_pairs(n: int = 2):
    """Load BoolQ pairs using the actual wisent extractor."""
    print("Loading BoolQ task via lm-eval...")
    task = LMEvalDataLoader.load_lm_eval_task("boolq")

    print("Extracting contrastive pairs...")
    extractor = BoolQExtractor()
    pairs = extractor.extract_contrastive_pairs(task, limit=n)

    return pairs


def show_raw_text(pair):
    """Show raw text without any template."""
    prompt = pair.prompt
    pos = pair.positive_response.model_response
    neg = pair.negative_response.model_response

    print("=" * 60)
    print("RAW TEXT (no template)")
    print("=" * 60)
    print(f"\n[PROMPT]:\n{prompt}")
    print(f"\n[POSITIVE RESPONSE]: {pos}")
    print(f"[NEGATIVE RESPONSE]: {neg}")
    print()
    print("Full positive text (simple concatenation):")
    full_pos = f"{prompt} {pos}"
    print(f"  '{full_pos}'")
    print()
    print("Full negative text (simple concatenation):")
    full_neg = f"{prompt} {neg}"
    print(f"  '{full_neg}'")
    print()


def show_extraction_strategy(pair, tokenizer, model_name: str, strategy: ExtractionStrategy):
    """Show text formatted by extraction strategy."""
    prompt = pair.prompt
    pos = pair.positive_response.model_response
    neg = pair.negative_response.model_response

    print("=" * 60)
    print(f"STRATEGY: {strategy.value} ({model_name})")
    print("=" * 60)

    try:
        # Build texts for positive response
        full_text_pos, answer_text_pos, prompt_only_pos = build_extraction_texts(
            strategy=strategy,
            prompt=prompt,
            response=pos,
            tokenizer=tokenizer,
            other_response=neg,
            is_positive=True,
            auto_convert_strategy=False,
        )

        # Build texts for negative response
        full_text_neg, answer_text_neg, prompt_only_neg = build_extraction_texts(
            strategy=strategy,
            prompt=prompt,
            response=neg,
            tokenizer=tokenizer,
            other_response=pos,
            is_positive=False,
            auto_convert_strategy=False,
        )

        print("\n[PROMPT ONLY]:")
        print("-" * 40)
        print(prompt_only_pos)

        print("\n[FULL POSITIVE]:")
        print("-" * 40)
        print(full_text_pos)

        print("\n[FULL NEGATIVE]:")
        print("-" * 40)
        print(full_text_neg)

    except Exception as e:
        print(f"  Error: {e}")

    print()


def test_chat_model(pairs):
    """Test chat extraction strategies on instruct/chat model."""
    strategies = [
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.MC_BALANCED,
    ]

    print("\n" + "=" * 70)
    print(f"LOADING CHAT MODEL: {CHAT_MODEL}")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)

    for i, pair in enumerate(pairs):
        print(f"\n{'#' * 70}")
        print(f"# PAIR {i + 1}")
        print(f"{'#' * 70}")

        show_raw_text(pair)

        for strategy in strategies:
            show_extraction_strategy(pair, tokenizer, CHAT_MODEL, strategy)

    del tokenizer
    torch.cuda.empty_cache()


def test_base_model(pairs):
    """Test extraction strategies on base model."""
    strategies = [
        ExtractionStrategy.COMPLETION_LAST,
        ExtractionStrategy.MC_COMPLETION,
    ]

    print("\n" + "=" * 70)
    print(f"LOADING BASE MODEL: {BASE_MODEL}")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    for i, pair in enumerate(pairs):
        print(f"\n{'#' * 70}")
        print(f"# PAIR {i + 1} (BASE MODEL)")
        print(f"{'#' * 70}")

        for strategy in strategies:
            show_extraction_strategy(pair, tokenizer, BASE_MODEL, strategy)

    del tokenizer
    torch.cuda.empty_cache()


def test_generation(pairs, model: WisentModel):
    """Test generation on BoolQ questions using WisentModel."""
    print("\n" + "=" * 70)
    print("TESTING GENERATION")
    print("=" * 70)

    for i, pair in enumerate(pairs):
        print(f"\n{'#' * 70}")
        print(f"# PAIR {i + 1} - GENERATION")
        print(f"{'#' * 70}")

        prompt = pair.prompt
        pos = pair.positive_response.model_response
        neg = pair.negative_response.model_response

        print(f"\n[PROMPT]:\n{prompt}")
        print(f"\n[EXPECTED POSITIVE]: {pos}")
        print(f"[EXPECTED NEGATIVE]: {neg}")

        # Generate response
        messages = [[{"role": "user", "content": prompt}]]
        responses = model.generate(
            inputs=messages,
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        )

        print(f"\n[GENERATED RESPONSE]: {responses[0]}")
        print()


def main():
    print("\n" + "#" * 70)
    print("# TEST: Extraction Strategy Text Formatting")
    print("#" * 70)

    # Load BoolQ pairs using wisent extractor
    print(f"\nLoading {N_SAMPLES} BoolQ pairs using BoolQExtractor...")
    pairs = load_boolq_pairs(N_SAMPLES)
    print(f"Loaded {len(pairs)} pairs\n")

    # Test apply chat template
    print(f"Testing apply_chat_template on {CHAT_MODEL}")
    test_chat_model(pairs)

    # Create model once
    print(f"Loading model: {CHAT_MODEL}")
    model = WisentModel(model_name=CHAT_MODEL, device=DEVICE)

    test_generation(pairs, model)

    del model
    torch.cuda.empty_cache()

    print("\n" + "#" * 70)
    print("# DONE")
    print("#" * 70)


if __name__ == "__main__":
    main()
