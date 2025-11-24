from datasets import load_dataset
import json

# Load concode dataset
print("Loading concode dataset...")

# Try different splits to see which ones exist
docs = None
split_used = None

for split in ['test', 'validation', 'train']:
    try:
        dataset = load_dataset("code_x_glue_tc_text_to_code", "default", split=split)
        docs = list(dataset)
        if docs:
            split_used = split
            break
    except:
        pass

if not docs:
    print("❌ No data found in any split!")
    exit(1)

print(f"Concode dataset loaded: {len(docs)} samples from {split_used}")
print(f"First sample keys: {list(docs[0].keys())}")
print("\n" + "="*80 + "\n")

# Display first 5 examples
for i, doc in enumerate(docs[:5]):
    print(f"EXAMPLE {i+1}:")

    # Extract specific fields
    nl = doc.get('nl')
    code = doc.get('code')

    # Print with types
    print(f"nl (natural language): {nl}")
    print(f"  type: {type(nl)}")
    print(f"\ncode: {code}")
    print(f"  type: {type(code)}")

    print("\n" + "="*80 + "\n")
