from datasets import load_dataset

# Load the dataset directly
dataset = load_dataset("CM/codexglue_code2text_go", split="test", trust_remote_code=True)

# Get first doc
if len(dataset) > 0:
    doc = dataset[0]
    print("First doc keys:", doc.keys())
    print("\nFirst doc:")
    print("code_tokens (first 20):", doc["code_tokens"][:20])
    print("docstring_tokens:", doc["docstring_tokens"])
    print("\nFull doc:", doc)
else:
    print("No docs found")
