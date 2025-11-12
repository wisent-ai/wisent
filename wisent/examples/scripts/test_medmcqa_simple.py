"""Simple test for medmcqa extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.medmcqa import MedmcqaExtractor

# Load dataset directly
print("Loading medmcqa dataset...")
dataset = load_dataset("medmcqa", split="validation")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MedmcqaExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Question: {doc.get('question', 'N/A')[:100]}...")
    print(f"Options: A={doc.get('opa', 'N/A')[:50]}..., B={doc.get('opb', 'N/A')[:50]}...")
    print(f"Correct: {doc.get('cop', 'N/A')}")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
