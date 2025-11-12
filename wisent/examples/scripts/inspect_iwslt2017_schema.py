from lm_eval import tasks
from lm_eval.tasks import TaskManager

task_manager = TaskManager()

# Check if iwslt2017 is a group or task
try:
    task_dict = task_manager.load_task_or_group("iwslt2017")
    print(f"Loaded iwslt2017: {type(task_dict)}")
    print(f"Number of items: {len(task_dict)}")

    # List all keys
    if len(task_dict) > 1:
        print(f"\nAll keys: {list(task_dict.keys())}")

    # Check first item
    first_key = list(task_dict.keys())[0]
    first_value = task_dict[first_key]
    print(f"\nFirst key: {first_key}")
    print(f"First value type: {type(first_value)}")

    # If it's a task object, try to get docs
    if hasattr(first_value, 'test_docs'):
        print("\nTrying to get docs...")
        for doc_source in ["test_docs", "validation_docs", "train_docs"]:
            try:
                docs_iter = getattr(first_value, doc_source)()
                if docs_iter is not None:
                    docs = list(docs_iter)
                    if docs:
                        print(f"Using {doc_source}()")
                        print("First doc keys:", docs[0].keys())
                        print("\nFirst doc:", docs[0])
                        break
            except:
                continue
except Exception as e:
    print(f"Error: {e}")
