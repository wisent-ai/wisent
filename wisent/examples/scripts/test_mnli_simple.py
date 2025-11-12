"""Simple test for mnli extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mnli import MnliExtractor

# Load dataset directly (test with validation_matched)
print("Loading mnli dataset (validation_matched)...")
dataset = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MnliExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Premise: {doc.get('premise', 'N/A')[:100]}...")
    print(f"Hypothesis: {doc.get('hypothesis', 'N/A')[:100]}...")
    print(f"Label: {doc.get('label', 'N/A')}")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
