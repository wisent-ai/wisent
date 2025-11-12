"""Simple test for okapi_arc_multilingual extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.okapi_arc_multilingual import OkapiArcMultilingualExtractor

# Load dataset directly (test with Spanish variant)
print("Loading okapi arc_es dataset...")
dataset = load_dataset("alexandrainst/m_arc", "es", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = OkapiArcMultilingualExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Query: {doc.get('query', 'N/A')[:100]}...")
    print(f"Gold: {doc.get('gold', 'N/A')}")
    print(f"Choices: {doc.get('choices', 'N/A')[:3] if isinstance(doc.get('choices'), list) else 'N/A'}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
