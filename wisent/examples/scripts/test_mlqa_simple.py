"""Simple test for mlqa extractor."""
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mlqa import MlqaExtractor

# Load dataset directly (test with English-English)
print("Loading mlqa dataset (English-English)...")
dataset = load_dataset("facebook/mlqa", "mlqa.en.en", split="test", trust_remote_code=True)
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MlqaExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Context: {doc.get('context', 'N/A')[:100]}...")
    print(f"Question: {doc.get('question', 'N/A')[:100]}...")
    print(f"Answers: {doc.get('answers', {})}")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
