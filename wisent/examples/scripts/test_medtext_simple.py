"""Simple test for medtext extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.medtext import MedtextExtractor

# Load dataset directly
print("Loading medtext dataset...")
dataset = load_dataset("BI55/MedText", split="train")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MedtextExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Prompt: {doc.get('Prompt', 'N/A')[:100]}...")
    print(f"Completion: {doc.get('Completion', 'N/A')[:100]}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
