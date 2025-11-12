"""Simple test for mmlusr extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mmlusr import MmlusrExtractor

# Load dataset directly (test with question_and_answer)
print("Loading mmlusr dataset (question_and_answer)...")
dataset = load_dataset("NiniCat/MMLU-SR", "question_and_answer", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MmlusrExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Doc keys: {list(doc.keys())}")
    print(f"Doc: {doc}")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
