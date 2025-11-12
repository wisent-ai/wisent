"""Simple test for noreval extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.noreval import NorevalExtractor

# Load dataset directly (test with norbelebele - Norwegian Bokmål Belebele)
print("Loading norbelebele dataset...")
dataset = load_dataset("facebook/belebele", "nob_Latn", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = NorevalExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Question: {doc.get('question', 'N/A')[:100]}...")
    print(f"Correct answer num: {doc.get('correct_answer_num', 'N/A')}")
    print(f"Answer 1: {doc.get('mc_answer1', 'N/A')[:50]}...")
    print(f"Answer 2: {doc.get('mc_answer2', 'N/A')[:50]}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
