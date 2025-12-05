"""
Run PolyMathEvaluator on PolyMath benchmark.

This script:
1. Loads Qwen/PolyMath dataset
2. Prompts an LLM to solve problems with language-specific instruction
3. Uses PolyMathEvaluator to compare model answer with ground truth
4. Computes DW-ACC (Difficulty-Weighted Accuracy) across all difficulty levels
5. Saves results to JSON file
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.benchmark_specific.polymath_evaluator import PolyMathEvaluator


# Generation configs following PolyMath benchmark methodology
# See: https://github.com/QwenLM/PolyMath
NON_REASONING_CONFIG = {
    "max_new_tokens": 65536,
    "temperature": 0.0,
    "do_sample": False,
}

REASONING_CONFIG = {
    "max_new_tokens": 65536,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "do_sample": True,
}

# Difficulty weights for DW-ACC metric
# Weights double at each level: solving 1 top = solving 8 low problems
DIFFICULTY_WEIGHTS = {"low": 1, "medium": 2, "high": 4, "top": 8}
TOTAL_WEIGHT = sum(DIFFICULTY_WEIGHTS.values())  # 15
DIFFICULTIES = ["low", "medium", "high", "top"]


def compute_dw_acc(accuracies: dict[str, float]) -> float:
    """
    Compute Difficulty-Weighted Accuracy (DW-ACC).

    DW-ACC assigns higher weights to harder problems:
    - low: 1, medium: 2, high: 4, top: 8

    Formula: DW-ACC = (1*a_low + 2*a_medium + 4*a_high + 8*a_top) / 15

    Args:
        accuracies: Dict mapping difficulty level to accuracy (0.0 to 1.0)
                    e.g., {"low": 0.8, "medium": 0.6, "high": 0.4, "top": 0.2}

    Returns:
        DW-ACC score (0.0 to 1.0)
    """
    weighted_sum = sum(
        DIFFICULTY_WEIGHTS[level] * acc
        for level, acc in accuracies.items()
    )
    return weighted_sum / TOTAL_WEIGHT


def compute_average_at_k(correct_counts: list[int], total: int = 125, k: int = 16) -> float:
    """
    Compute average@k accuracy for reasoning models.

    Reasoning models use sampling (T=0.6), so results vary between runs.
    This function averages results across k trials with rounding to preserve
    granularity (1/total per problem).

    Formula: average@k = round(sum(correct_counts) / k) / total

    Args:
        correct_counts: List of correct answers per trial, e.g., [102, 98, 105, ...]
        total: Total problems per level (default 125 for PolyMath)
        k: Number of trials (default 16)

    Returns:
        Accuracy as float (0.0 to 1.0)

    Note:
        This function is for reasoning models only. For non-reasoning models,
        use greedy decoding (T=0.0) and run once per problem.
    """
    avg_correct = sum(correct_counts) / k
    return round(avg_correct) / total


def evaluate_difficulty(
    model: WisentModel,
    evaluator: PolyMathEvaluator,
    language: str,
    difficulty: str,
    limit: int | None = None,
) -> tuple[float, int, list[dict]]:
    """Evaluate model on a single difficulty level.

    Returns:
        Tuple of (accuracy, total_count, results_list)
    """
    ds = load_dataset("Qwen/PolyMath", language, split=difficulty)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    results = []

    for example in tqdm(ds, desc=f"Evaluating {difficulty}"):
        question = example.get('question', '')
        answer = example.get('answer', '')

        prompt = PolyMathEvaluator.get_prompt(question, language)

        responses = model.generate(
            inputs=prompt,
            **NON_REASONING_CONFIG,
            prompt_is_formatted=True,
        )

        response = responses[0] if responses else ""
        eval_result = evaluator.evaluate(response, answer)

        if eval_result.ground_truth == "TRUTHFUL":
            correct += 1

        results.append({
            'question': question,
            'true_answer': answer,
            'model_output': response,
            'ground_truth': eval_result.ground_truth,
            'confidence': eval_result.confidence,
            'details': eval_result.details,
        })

    accuracy = correct / len(ds) if len(ds) > 0 else 0.0
    return accuracy, len(ds), results


def main(limit: int | None = None, language: str = "en"):
    """Run evaluation on all difficulty levels and compute DW-ACC."""
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = WisentModel(model_name=model_name)

    evaluator = PolyMathEvaluator()

    accuracies = {}
    all_results = {}
    total_correct = 0
    total_count = 0

    print(f"\nEvaluating on language: {language}")
    print("=" * 60)

    for difficulty in DIFFICULTIES:
        print(f"\n--- {difficulty.upper()} ---")
        accuracy, count, results = evaluate_difficulty(
            model, evaluator, language, difficulty, limit
        )
        accuracies[difficulty] = accuracy
        all_results[difficulty] = results
        total_correct += int(accuracy * count)
        total_count += count

        print(f"{difficulty}: {accuracy*100:.2f}% ({int(accuracy*count)}/{count})")

    # Compute DW-ACC
    dw_acc = compute_dw_acc(accuracies)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Language: {language}")
    print(f"Model: {model_name}")
    print()
    for difficulty in DIFFICULTIES:
        weight = DIFFICULTY_WEIGHTS[difficulty]
        acc = accuracies[difficulty]
        print(f"  {difficulty:6s}: {acc*100:5.2f}% (weight={weight})")
    print()
    print(f"Overall Accuracy: {total_correct/total_count*100:.2f}% ({total_correct}/{total_count})")
    print(f"DW-ACC: {dw_acc*100:.2f}%")

    # Save results to JSON
    output_dir = Path(__file__).parent / "results_test_evaluator"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"polymath_evaluator_results_{language}.json"

    output_data = {
        "model_name": model_name,
        "dataset": "Qwen/PolyMath",
        "language": language,
        "accuracies": {k: v * 100 for k, v in accuracies.items()},
        "dw_acc": dw_acc * 100,
        "overall_accuracy": total_correct / total_count * 100,
        "total_correct": total_correct,
        "total_count": total_count,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main(limit=2, language="en")
