"""Test evaluation for mlqa benchmark."""
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mlqa import MlqaExtractor
from wisent.core.evaluators.benchmark_specific.generation_evaluator import GenerationEvaluator

# Load dataset directly (test with English-English)
print("Loading mlqa dataset (English-English)...")
dataset = load_dataset("facebook/mlqa", "mlqa.en.en", split="test", trust_remote_code=True)
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MlqaExtractor()

# Extract first pair
print("\nExtracting first pair...")
docs = list(dataset)[:1]
pair = extractor._extract_pair_from_doc(docs[0])

if not pair:
    print("✗ Failed to extract pair")
    exit(1)

print(f"✓ Successfully extracted pair")
print(f"  Prompt: {pair.prompt[:150]}...")
print(f"  Positive: {pair.positive_response.model_response[:100]}...")
print(f"  Negative: {pair.negative_response.model_response[:100]}...")

# Test evaluation
print("\nTesting evaluation...")
evaluator = GenerationEvaluator()

# Test 1: Evaluate with choices (contrastive evaluation)
print("\n--- Test 1: Contrastive evaluation with correct answer ---")
result = evaluator.evaluate(
    response="",  # Not used in contrastive mode
    expected=pair.positive_response.model_response,
    choices=[pair.positive_response.model_response, pair.negative_response.model_response],
    task_name="mlqa",
    answer_type="text"
)

print(f"Ground truth: {result.ground_truth}")
print(f"Confidence: {result.confidence}")
print(f"Details: {result.details}")

if result.ground_truth != "TRUTHFUL":
    print("✗ FAILED: Expected TRUTHFUL when correct answer matches expected")
    exit(1)
else:
    print("✓ PASSED: Correct answer recognized as truthful")

# Test 2: Evaluate with incorrect answer as expected
print("\n--- Test 2: Contrastive evaluation with incorrect answer ---")
result2 = evaluator.evaluate(
    response="",
    expected=pair.negative_response.model_response,
    choices=[pair.positive_response.model_response, pair.negative_response.model_response],
    task_name="mlqa",
    answer_type="text"
)

print(f"Ground truth: {result2.ground_truth}")
print(f"Confidence: {result2.confidence}")
print(f"Details: {result2.details}")

if result2.ground_truth != "UNTRUTHFUL":
    print("✗ FAILED: Expected UNTRUTHFUL when incorrect answer matches expected")
    exit(1)
else:
    print("✓ PASSED: Incorrect answer recognized as untruthful")

print("\n✓ All evaluation tests passed!")
print("✓ mlqa benchmark is FULLY VERIFIED (extraction + evaluation)")
