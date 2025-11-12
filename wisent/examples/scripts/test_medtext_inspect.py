"""Inspect medtext pairs to understand the content."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.medtext import MedtextExtractor

# Load dataset directly
print("Loading medtext dataset...")
dataset = load_dataset("BI55/MedText", split="train")
print(f"Loaded {len(dataset)} examples\n")

# Create extractor
extractor = MedtextExtractor()

# Extract first pair and inspect it
docs = list(dataset)[:1]

print("="*80)
print("DOCUMENT 1")
print("="*80)

pair = extractor._extract_pair_from_doc(docs[0])

if pair:
    print(f"\nPROMPT:\n{pair.prompt}\n")
    print(f"POSITIVE RESPONSE:\n{pair.positive_response.model_response}\n")
    print(f"NEGATIVE RESPONSE:\n{pair.negative_response.model_response}\n")

    print(f"POSITIVE LENGTH: {len(pair.positive_response.model_response)} chars")
    print(f"NEGATIVE LENGTH: {len(pair.negative_response.model_response)} chars")
    print(f"ARE THEY EQUAL? {pair.positive_response.model_response == pair.negative_response.model_response}")
else:
    print("Failed to extract pair")
