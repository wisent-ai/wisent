"""Simple test for mimic_repsum extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mimic_repsum import MimicRepsumExtractor

# Load dataset directly
print("Loading mimic_repsum dataset...")
dataset = load_dataset("dmacres/mimiciii-hospitalcourse-meta", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MimicRepsumExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    text = doc.get('extractive_notes_summ', 'N/A')
    print(f"Text (first 100 chars): {text[:100]}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
