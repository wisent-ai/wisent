from lm_eval.tasks import get_task_dict
import json

# Load BoolQ task
tasks = get_task_dict(["boolq"])
boolq_task = tasks["boolq"]

# Try different splits to see which ones exist
docs = None
split_used = None

for split_method in ['test_docs', 'validation_docs', 'train_docs', 'fewshot_docs']:
    if hasattr(boolq_task, split_method):
        try:
            docs = list(getattr(boolq_task, split_method)())
            if docs:
                split_used = split_method
                break
        except:
            pass

if not docs:
    print("❌ No data found in any split!")
    exit(1)

print(f"BoolQ dataset loaded: {len(docs)} samples from {split_used}")
print(f"First sample keys: {list(docs[0].keys())}")
print("\n" + "="*80 + "\n")

# Display first 5 examples
for i, doc in enumerate(docs[:5]):
    print(f"EXAMPLE {i+1}:")

    # Extract specific fields
    question = doc.get('question')
    passage = doc.get('passage')
    label = doc.get('label')
    idx = doc.get('idx')

    # Print with types
    print(f"question: {question}")
    print(f"  type: {type(question)}")
    print(f"\npassage: {passage}")
    print(f"  type: {type(passage)}")
    print(f"\nlabel: {label}")
    print(f"  type: {type(label)}")
    print(f"\nidx: {idx}")
    print(f"  type: {type(idx)}")

    print("\n" + "="*80 + "\n")