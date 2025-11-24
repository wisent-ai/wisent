from lm_eval.tasks import get_task_dict
import json

# Load SQuAD2 task
tasks = get_task_dict(["squadv2"])
squad2_task = tasks["squadv2"]

# Try different splits to see which ones exist
docs = None
split_used = None

for split_method in ['test_docs', 'validation_docs', 'train_docs', 'fewshot_docs']:
    if hasattr(squad2_task, split_method):
        try:
            docs = list(getattr(squad2_task, split_method)())
            if docs:
                split_used = split_method
                break
        except:
            pass

if not docs:
    print("❌ No data found in any split!")
    exit(1)

print(f"SQuAD2 dataset loaded: {len(docs)} samples from {split_used}")
print(f"First sample keys: {list(docs[0].keys())}")
print("\n" + "="*80 + "\n")

# Display first 5 examples
for i, doc in enumerate(docs[:5]):
    print(f"EXAMPLE {i+1}:")
    print(json.dumps(doc, indent=2, default=str))
    print("\n" + "="*80 + "\n")
