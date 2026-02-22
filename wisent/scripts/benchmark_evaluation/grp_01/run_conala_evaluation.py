"""
Run CoNaLaEvaluator on CoNaLa benchmark.

This script:
1. Loads neulab/conala dataset (curated split)
2. Prompts an LLM to generate Python code from natural language intent
3. Computes corpus-level BLEU score following official CoNaLa baseline
4. Saves results to JSON file

Dataset: https://huggingface.co/datasets/neulab/conala
Baseline: https://github.com/conala-corpus/conala-baseline/
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.benchmark_specific.conala_evaluator import (
    CoNaLaEvaluator,
    tokenize_for_bleu_eval,
)
from wisent.core.evaluators.benchmark_specific.math_parsing.extract_boxed import extract_boxed_answer


# Generation config for code generation
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.0,
    "do_sample": False,
}


def evaluate_conala(
    model: WisentModel,
    evaluator: CoNaLaEvaluator,
    split: str = "test",
    limit: int | None = None,
) -> dict:
    """Evaluate model on CoNaLa dataset.

    Args:
        model: WisentModel instance
        evaluator: CoNaLaEvaluator instance
        split: Dataset split ('train' or 'test')
        limit: Optional limit on number of examples

    Returns:
        Dictionary with BLEU score, exact match, and detailed results
    """

    ds = load_dataset("neulab/conala", "curated", split=split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    generated_responses = []
    expected_snippets = []
    exact_matches = 0

    for example in tqdm(ds, desc=f"Evaluating {split}"):
        rewritten_intent = example.get('rewritten_intent', '')
        snippet = example.get('snippet', '')

        # Skip if no rewritten_intent
        if not rewritten_intent or not snippet:
            continue

        prompt = evaluator.get_prompt(rewritten_intent)

        responses = model.generate(
            inputs=prompt,
            **GENERATION_CONFIG,
            prompt_is_formatted=True,
        )

        raw_response = responses[0] if responses else ""

        # Extract code from \boxed{}
        response = extract_boxed_answer(raw_response)
        if response is None:
            response = ""  # No boxed answer found

        generated_responses.append(response)
        expected_snippets.append(snippet)

        # Check exact match (after tokenization)
        ref_tokens = tokenize_for_bleu_eval(snippet)
        hyp_tokens = tokenize_for_bleu_eval(response)
        is_exact_match = ref_tokens == hyp_tokens
        if is_exact_match:
            exact_matches += 1

        results.append({
            'rewritten_intent': rewritten_intent,
            'reference_snippet': snippet,
            'raw_output': raw_response,
            'extracted_code': response,
            'exact_match': is_exact_match,
        })

    # Compute corpus-level BLEU using evaluator
    corpus_metrics = evaluator.evaluate_corpus(generated_responses, expected_snippets)

    return {
        'bleu_score': corpus_metrics['bleu_score'],
        'exact_match': exact_matches / len(results) * 100 if results else 0.0,
        'exact_match_count': exact_matches,
        'total': len(results),
        'precisions': corpus_metrics['precisions'],
        'brevity_penalty': corpus_metrics['brevity_penalty'],
        'length_ratio': corpus_metrics['length_ratio'],
        'results': results,
    }


def main(limit: int | None = None, split: str = "test"):
    """Run CoNaLa evaluation and save results."""
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = WisentModel(model_name=model_name)

    evaluator = CoNaLaEvaluator()

    print(f"\nEvaluating on CoNaLa {split} split")
    print("=" * 60)

    metrics = evaluate_conala(model, evaluator, split, limit)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: neulab/conala (curated)")
    print(f"Split: {split}")
    print(f"Examples: {metrics['total']}")
    print()
    print(f"BLEU Score: {metrics['bleu_score']:.2f}")
    print(f"Exact Match: {metrics['exact_match']:.2f}% ({metrics['exact_match_count']}/{metrics['total']})")
    print(f"Brevity Penalty: {metrics['brevity_penalty']:.4f}")
    print(f"Length Ratio: {metrics['length_ratio']:.4f}")
    print()
    print("N-gram Precisions:")
    for i, p in enumerate(metrics['precisions'], 1):
        print(f"  {i}-gram: {p*100:.2f}%")

    # Save results to JSON
    output_dir = Path(__file__).parent / "results_test_evaluator"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"conala_evaluator_results_{split}.json"

    output_data = {
        "model_name": model_name,
        "dataset": "neulab/conala",
        "split": split,
        **metrics,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main(limit=2, split="test")
