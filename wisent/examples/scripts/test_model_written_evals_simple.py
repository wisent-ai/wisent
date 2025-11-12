"""Simple test for model_written_evals extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.model_written_evals import ModelWrittenEvalsExtractor

# Load dataset directly (test with advanced_ai_risk)
print("Loading advanced_ai_risk dataset (human-power-seeking-inclination)...")
dataset = load_dataset("EleutherAI/advanced_ai_risk", "human-power-seeking-inclination", split="validation")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = ModelWrittenEvalsExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Question: {doc.get('question', 'N/A')[:100]}...")
    print(f"Answer matching behavior: {doc.get('answer_matching_behavior', 'N/A')[:50]}...")
    print(f"Answer not matching behavior: {doc.get('answer_not_matching_behavior', 'N/A')[:50]}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
