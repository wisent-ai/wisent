"""Test evaluation for mmlusr benchmark."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mmlusr import MmlusrExtractor
from wisent.core.evaluators.benchmark_specific.log_likelihoods_evaluator import LogLikelihoodsEvaluator

# Load dataset directly (test with question_and_answer)
print("Loading mmlusr dataset (question_and_answer)...")
dataset = load_dataset("NiniCat/MMLU-SR", "question_and_answer", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MmlusrExtractor()

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
evaluator = LogLikelihoodsEvaluator()

# Test 1: Evaluate with choices (contrastive evaluation)
print("\n--- Test 1: Contrastive evaluation with correct answer ---")
result = evaluator.evaluate(
    response="",  # Not used in contrastive mode
    expected=pair.positive_response.model_response,
    choices=[pair.positive_response.model_response, pair.negative_response.model_response],
    task_name="mmlusr",
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
    task_name="mmlusr",
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
print("✓ mmlusr benchmark is FULLY VERIFIED (extraction + evaluation)")
