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
    compute_bleu,
)
from wisent.core.evaluators.benchmark_specific.utils import extract_boxed_answer


# Generation config for code generation
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.0,
    "do_sample": False,
}


def get_few_shot_examples(n: int = 2) -> list[tuple[str, str]]:
    """Load few-shot examples from training split.

    Args:
        n: Number of examples to load

    Returns:
        List of (intent, snippet) tuples
    """
    train_ds = load_dataset("neulab/conala", "curated", split="train")
    examples = []
    for i in range(min(n, len(train_ds))):
        ex = train_ds[i]
        # Use rewritten_intent if available
        intent = ex.get('rewritten_intent') or ex.get('intent', '')
        snippet = ex.get('snippet', '')
        examples.append((intent, snippet))
    return examples


def evaluate_conala(
    model: WisentModel,
    evaluator: CoNaLaEvaluator,
    split: str = "test",
    limit: int | None = None,
    num_examples: int = 2,
) -> dict:
    """Evaluate model on CoNaLa dataset.

    Args:
        model: WisentModel instance
        evaluator: CoNaLaEvaluator instance
        split: Dataset split ('train' or 'test')
        limit: Optional limit on number of examples
        num_examples: Number of few-shot examples from training

    Returns:
        Dictionary with BLEU score, exact match, and detailed results
    """
    # Load few-shot examples from training
    few_shot_examples = get_few_shot_examples(num_examples)

    ds = load_dataset("neulab/conala", "curated", split=split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    references = []
    hypotheses = []
    exact_matches = 0

    for example in tqdm(ds, desc=f"Evaluating {split}"):
        intent = example.get('intent', '')
        rewritten_intent = example.get('rewritten_intent', '')
        snippet = example.get('snippet', '')

        # Use rewritten_intent if available (better quality)
        prompt = evaluator.get_prompt(intent, rewritten_intent, examples=few_shot_examples)

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

        # Tokenize for BLEU
        ref_tokens = tokenize_for_bleu_eval(snippet)
        hyp_tokens = tokenize_for_bleu_eval(response)

        references.append(ref_tokens)
        hypotheses.append(hyp_tokens)

        # Check exact match (after tokenization)
        is_exact_match = ref_tokens == hyp_tokens
        if is_exact_match:
            exact_matches += 1

        results.append({
            'intent': intent,
            'rewritten_intent': rewritten_intent,
            'reference_snippet': snippet,
            'raw_output': raw_response,
            'extracted_code': response,
            'exact_match': is_exact_match,
        })

    # Compute corpus-level BLEU
    bleu, precisions, bp, ratio, _, _ = compute_bleu(references, hypotheses)

    return {
        'bleu_score': bleu * 100,
        'exact_match': exact_matches / len(results) * 100 if results else 0.0,
        'exact_match_count': exact_matches,
        'total': len(results),
        'precisions': precisions,
        'brevity_penalty': bp,
        'length_ratio': ratio,
        'results': results,
    }


def main(limit: int | None = None, split: str = "test"):
    """Run CoNaLa evaluation and save results."""
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
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
    main(limit=1, split="test")
