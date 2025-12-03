"""Test GenerationEvaluator answer extraction on competition_math dataset."""

from datasets import load_dataset
from wisent.core.evaluators.benchmark_specific.generation_evaluator import GenerationEvaluator

# Load dataset
ds = load_dataset("qwedsacf/competition_math", split="train")

# Initialize evaluator
evaluator = GenerationEvaluator()

# Test on a few examples
num_examples = 5

print("=" * 80)
print("Testing GenerationEvaluator answer extraction on competition_math")
print("=" * 80)

for i in range(num_examples):
    example = ds[i]
    solution = example["solution"]

    print(f"\n--- Example {i} ---")
    print(f"Problem: {example['problem'][:100]}...")
    print(f"\nFull solution:\n{solution}")

    # Extract numerical answer
    extracted = evaluator._extract_numerical_answer(solution)
    print(f"\n>>> Extracted numerical answer: {extracted}")

    # Also try text extraction for comparison
    text_extracted = evaluator._extract_text_answer(solution)
    print(f">>> Extracted text answer: {text_extracted[:100]}..." if len(str(text_extracted)) > 100 else f">>> Extracted text answer: {text_extracted}")

    print("-" * 80)
