"""Check olaph dataset fields."""
from datasets import load_dataset

# Load dataset directly
print("Loading olaph dataset...")
dataset = load_dataset("dmis-lab/MedLFQA", split="test")
print(f"Loaded {len(dataset)} examples")

# Check first document fields
if len(dataset) > 0:
    first_doc = dataset[0]
    print("\nFields in first document:")
    for key, value in first_doc.items():
        print(f"  {key}: {type(value).__name__} = {str(value)[:100]}...")
