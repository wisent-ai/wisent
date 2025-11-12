"""Inspect mediqa_qa2019 pairs to understand the content."""
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mediqa_qa2019 import MediqaQa2019Extractor

# Load dataset directly
print("Loading mediqa_qa dataset...")
dataset = load_dataset("bigbio/mediqa_qa", "mediqa_qa_source", split="train_live_qa_med")
print(f"Loaded {len(dataset)} examples\n")

# Create extractor
extractor = MediqaQa2019Extractor()

# Extract first 3 pairs and inspect them
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n{'='*80}")
    print(f"DOCUMENT {i+1}")
    print(f"{'='*80}")

    pair = extractor._extract_pair_from_doc(doc)

    if pair:
        print(f"\nPROMPT:\n{pair.prompt}\n")
        print(f"POSITIVE RESPONSE:\n{pair.positive_response.model_response}\n")
        print(f"NEGATIVE RESPONSE:\n{pair.negative_response.model_response}\n")
    else:
        print("Failed to extract pair")
