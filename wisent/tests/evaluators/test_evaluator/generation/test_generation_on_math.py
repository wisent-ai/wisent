"""Test GenerationEvaluator answer extraction on competition_math dataset."""

import argparse
from datasets import load_dataset
from wisent.core.reading.evaluators.benchmark_specific.generation_evaluator import GenerationEvaluator
from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_COMPACT, SEPARATOR_WIDTH_REPORT

_parser = argparse.ArgumentParser()
_parser.add_argument("--generation-embedding-weight", type=float, required=True)
_parser.add_argument("--generation-nli-weight", type=float, required=True)
_args = _parser.parse_args()

# Load dataset
ds = load_dataset("qwedsacf/competition_math", split="train")

# Initialize evaluator
evaluator = GenerationEvaluator(
    generation_embedding_weight=_args.generation_embedding_weight,
    generation_nli_weight=_args.generation_nli_weight,
)

# Test on a few examples
num_examples = 5

print("=" * SEPARATOR_WIDTH_REPORT)
print("Testing GenerationEvaluator answer extraction on competition_math")
print("=" * SEPARATOR_WIDTH_REPORT)

for i in range(num_examples):
    example = ds[i]
    solution = example["solution"]

    print(f"\n--- Example {i} ---")
    print(f"Problem: {example['problem'][:DISPLAY_TRUNCATION_COMPACT]}...")
    print(f"\nFull solution:\n{solution}")

    # Extract numerical answer
    extracted = evaluator._extract_numerical_answer(solution)
    print(f"\n>>> Extracted numerical answer: {extracted}")

    # Also try text extraction for comparison
    text_extracted = evaluator._extract_text_answer(solution)
    print(f">>> Extracted text answer: {text_extracted[:DISPLAY_TRUNCATION_COMPACT]}..." if len(str(text_extracted)) > DISPLAY_TRUNCATION_COMPACT else f">>> Extracted text answer: {text_extracted}")

    print("-" * SEPARATOR_WIDTH_REPORT)
