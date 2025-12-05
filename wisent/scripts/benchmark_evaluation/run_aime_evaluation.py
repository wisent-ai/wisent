"""
Run AIMEEvaluator on AIME benchmark.

This script:
1. Loads gneubig/aime-1983-2024 dataset
2. Prompts an LLM to solve problems and put answer in \\boxed{}
3. Uses AIMEEvaluator to compare model answer with ground truth
4. Saves results to JSON file
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.benchmark_specific.aime_evaluator import AIMEEvaluator


def main(limit: int | None = None):
    # Load model using WisentModel
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = WisentModel(model_name=model_name)

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("gneubig/aime-1983-2024", split='train')

    # Apply limit if specified
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Initialize evaluator
    evaluator = AIMEEvaluator()

    correct = 0
    incorrect = 0
    unknown = 0

    results = []

    print(f"\nTesting on {len(ds)} examples...")
    print("=" * 60)

    for i, example in enumerate(tqdm(ds, desc="Evaluating")):
        problem = example.get('Question', '')
        answer = example.get('Answer', '')  # AIME answers are integers 0-999

        # Create prompt and generate response
        prompt = AIMEEvaluator.get_prompt(problem)

        # Generate using WisentModel
        responses = model.generate(
            inputs=prompt,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            prompt_is_formatted=True
        )

        response = responses[0] if responses else ""

        # Evaluate using AIMEEvaluator
        eval_result = evaluator.evaluate(response, answer)

        if eval_result.ground_truth == "TRUTHFUL":
            correct += 1
        elif eval_result.ground_truth == "UNTRUTHFUL":
            incorrect += 1
        else:
            unknown += 1

        results.append({
            'problem': problem,
            'true_answer': answer,
            'model_output': response,
            'ground_truth': eval_result.ground_truth,
            'confidence': eval_result.confidence,
            'details': eval_result.details,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total examples: {len(ds)}")
    print(f"Correct (TRUTHFUL): {correct}")
    print(f"Incorrect (UNTRUTHFUL): {incorrect}")
    print(f"Unknown: {unknown}")

    evaluated = correct + incorrect
    if evaluated > 0:
        accuracy = 100 * correct / evaluated
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        accuracy = 0.0

    # Save results to JSON
    output_dir = Path(__file__).parent / "results_test_evaluator"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "aime_evaluator_results.json"

    output_data = {
        "model_name": model_name,
        "dataset": "gneubig/aime-1983-2024",
        "total_examples": len(ds),
        "correct": correct,
        "incorrect": incorrect,
        "unknown": unknown,
        "accuracy": accuracy,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main(limit=50)
