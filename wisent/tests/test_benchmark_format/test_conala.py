from datasets import load_dataset
import json

# Load conala dataset
print("Loading conala dataset...")

# Try different splits to see which ones exist
docs = None
split_used = None

for split in ['test', 'validation', 'train']:
    try:
        dataset = load_dataset("neulab/conala", split=split)
        docs = list(dataset)
        if docs:
            split_used = split
            break
    except:
        pass

if not docs:
    print("❌ No data found in any split!")
    exit(1)

print(f"Conala dataset loaded: {len(docs)} samples from {split_used}")
print(f"First sample keys: {list(docs[0].keys())}")
print("\n" + "="*80 + "\n")

# Display first 5 examples
for i, doc in enumerate(docs[:5]):
    print(f"EXAMPLE {i+1}:")

    # Extract specific fields
    intent = doc.get('intent')
    snippet = doc.get('snippet')

    # Print with types
    print(f"intent: {intent}")
    print(f"  type: {type(intent)}")
    print(f"\nsnippet: {snippet}")
    print(f"  type: {type(snippet)}")

    print("\n" + "="*80 + "\n")
