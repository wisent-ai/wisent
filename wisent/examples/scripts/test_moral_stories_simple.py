"""Simple test for moral_stories extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.moral_stories import MoralStoriesExtractor

# Load dataset directly (test with full train split)
print("Loading moral_stories dataset (full)...")
dataset = load_dataset("demelin/moral_stories", "full", split="train")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MoralStoriesExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Norm: {doc.get('norm', 'N/A')[:50]}...")
    print(f"Situation: {doc.get('situation', 'N/A')[:50]}...")
    print(f"Intention: {doc.get('intention', 'N/A')[:50]}...")
    print(f"Moral action: {doc.get('moral_action', 'N/A')[:50]}...")
    print(f"Immoral action: {doc.get('immoral_action', 'N/A')[:50]}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
