"""Simple test for mgsm extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mgsm import MgsmExtractor

# Load dataset directly (test with English variant)
print("Loading mgsm dataset (English)...")
dataset = load_dataset("juletxara/mgsm", "en", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MgsmExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Question: {doc.get('question', 'N/A')[:100]}...")
    print(f"Answer number: {doc.get('answer_number', 'N/A')}")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
