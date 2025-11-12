"""
Direct test of code2text_go extractor using dataset loading.
"""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.code2text import Code2textExtractor

# Load the dataset directly
dataset = load_dataset("CM/codexglue_code2text_go", split="test", trust_remote_code=True)

# Create extractor
extractor = Code2textExtractor()

# Get first doc and extract pair
doc = dataset[0]
print("Testing document:")
print(f"  code_tokens (first 10): {doc['code_tokens'][:10]}")
print(f"  docstring_tokens: {doc['docstring_tokens']}")

pair = extractor._extract_pair_from_doc(doc)

if pair:
    print("\n✓ Pair extracted successfully!")
    print(f"\nPrompt (first 200 chars):\n{pair.prompt[:200]}...")
    print(f"\nPositive response:\n{pair.positive_response.model_response}")
    print(f"\nNegative response:\n{pair.negative_response.model_response}")
else:
    print("\n✗ Failed to extract pair")
