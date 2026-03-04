"""Test GenerationEvaluator answer extraction on AIME 2024 dataset."""

from datasets import load_dataset
from wisent.core.reading.evaluators.benchmark_specific.generation_evaluator import GenerationEvaluator
from wisent.core.utils.config_tools.constants import COMPARE_TOL, DISPLAY_TRUNCATION_COMPACT, SEPARATOR_WIDTH_REPORT, GENERATION_EMBEDDING_WEIGHT, GENERATION_NLI_WEIGHT

# Load dataset
ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

# Initialize evaluator
evaluator = GenerationEvaluator(
    generation_embedding_weight=GENERATION_EMBEDDING_WEIGHT,
    generation_nli_weight=GENERATION_NLI_WEIGHT,
)

# Test on a few examples
num_examples = 5

print("=" * SEPARATOR_WIDTH_REPORT)
print("Testing GenerationEvaluator answer extraction on AIME 2024")
print("=" * SEPARATOR_WIDTH_REPORT)

correct = 0
total = num_examples

for i in range(num_examples):
    example = ds[i]
    solution = example["solution"]
    expected_answer = example["answer"]

    print(f"\n--- Example {i} ---")
    print(f"Problem: {example['problem'][:DISPLAY_TRUNCATION_COMPACT]}...")
    print(f"\nFull solution:\n{solution}")
    print(f"\n>>> Expected answer: {expected_answer}")

    # Extract numerical answer
    extracted = evaluator._extract_numerical_answer(solution)
    print(f">>> Extracted numerical answer: {extracted}")

    # Check if correct
    try:
        is_correct = abs(float(extracted) - float(expected_answer)) < COMPARE_TOL
        if is_correct:
            correct += 1
            print(">>> ✓ CORRECT")
        else:
            print(">>> ✗ WRONG")
    except (TypeError, ValueError):
        print(">>> ✗ FAILED TO EXTRACT")

    print("-" * SEPARATOR_WIDTH_REPORT)

print(f"\n\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
