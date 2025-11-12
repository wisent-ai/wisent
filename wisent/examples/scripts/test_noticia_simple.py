"""Simple test for noticia extractor."""
from datasets import load_dataset
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.noticia import NoticiaExtractor

# Load dataset directly
print("Loading noticia dataset...")
dataset = load_dataset("Iker/NoticIA", split="test")
print(f"Loaded {len(dataset)} examples")

# Create extractor
extractor = NoticiaExtractor()

# Test extraction on first 3 documents
print("\nTesting extraction on first 3 documents...")
docs = list(dataset)[:3]

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Headline: {doc.get('web_headline', 'N/A')[:100]}...")
    print(f"Text: {doc.get('web_text', 'N/A')[:100]}...")
    print(f"Summary: {doc.get('summary', 'N/A')[:100]}...")

    pair = extractor._extract_pair_from_doc(doc)
    if pair:
        print(f"✓ Successfully extracted pair")
        print(f"  Prompt: {pair.prompt[:150]}...")
        print(f"  Positive: {pair.positive_response.model_response[:100]}...")
        print(f"  Negative: {pair.negative_response.model_response[:100]}...")
    else:
        print(f"✗ Failed to extract pair")

print("\n✓ Test completed successfully!")
