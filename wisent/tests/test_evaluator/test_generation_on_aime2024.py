"""Test GenerationEvaluator answer extraction on AIME 2024 dataset."""

from datasets import load_dataset
from wisent.core.evaluators.benchmark_specific.generation_evaluator import GenerationEvaluator

# Load dataset
ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

# Initialize evaluator
evaluator = GenerationEvaluator()

# Test on a few examples
num_examples = 5

print("=" * 80)
print("Testing GenerationEvaluator answer extraction on AIME 2024")
print("=" * 80)

correct = 0
total = num_examples

for i in range(num_examples):
    example = ds[i]
    solution = example["solution"]
    expected_answer = example["answer"]

    print(f"\n--- Example {i} ---")
    print(f"Problem: {example['problem'][:100]}...")
    print(f"\nFull solution:\n{solution}")
    print(f"\n>>> Expected answer: {expected_answer}")

    # Extract numerical answer
    extracted = evaluator._extract_numerical_answer(solution)
    print(f">>> Extracted numerical answer: {extracted}")

    # Check if correct
    try:
        is_correct = abs(float(extracted) - float(expected_answer)) < 1e-6
        if is_correct:
            correct += 1
            print(">>> ✓ CORRECT")
        else:
            print(">>> ✗ WRONG")
    except (TypeError, ValueError):
        print(">>> ✗ FAILED TO EXTRACT")

    print("-" * 80)

print(f"\n\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
