from lm_eval.tasks import get_task_dict

# Load arabic_exams task
tasks = get_task_dict(["arabic_exams"])
arabic_exams_task = tasks["arabic_exams"]

# Try different splits to see which ones exist
docs = None
split_used = None

for split_method in ['test_docs', 'validation_docs', 'train_docs', 'fewshot_docs']:
    if hasattr(arabic_exams_task, split_method):
        try:
            docs = list(getattr(arabic_exams_task, split_method)())
            if docs:
                split_used = split_method
                break
        except:
            pass

if not docs:
    print("❌ No data found in any split!")
    exit(1)

print(f"Arabic_EXAMS dataset loaded: {len(docs)} samples from {split_used}")
print(f"First sample keys: {list(docs[0].keys())}")
print("\n" + "="*80 + "\n")

# Display first 5 examples
for i, doc in enumerate(docs[:5]):
    print(f"EXAMPLE {i+1}:")
    print(f"All keys: {list(doc.keys())}")
    print()

    # Print all fields with their types and values
    for key, value in doc.items():
        # Truncate long values for readability
        if isinstance(value, str) and len(value) > 500:
            display_value = value[:500] + "..."
        else:
            display_value = value
        print(f"{key}: {display_value}")
        print(f"  type: {type(value)}")
        print()

    print("="*80 + "\n")
