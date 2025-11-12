"""Simple test for mediqa_qa2019 extractor that bypasses lm-eval's evaluation setup."""
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mediqa_qa2019 import MediqaQa2019Extractor

# Load dataset directly
print("Loading mediqa_qa dataset...")
dataset = load_dataset("bigbio/mediqa_qa", "mediqa_qa_source", split="train_live_qa_med")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = MediqaQa2019Extractor()

# Test extraction on first 2 documents
print("\nTesting extraction on first 2 documents...")
docs = list(dataset)[:2]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:100]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
