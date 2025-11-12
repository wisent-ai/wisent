from lm_eval import tasks
from lm_eval.tasks import TaskManager

task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("fda")
task_obj = list(task_dict.values())[0]

# Try different doc sources
for doc_source in ["test_docs", "validation_docs", "train_docs"]:
    try:
        docs_iter = getattr(task_obj, doc_source)()
        if docs_iter is not None:
            docs = list(docs_iter)
            if docs:
                print(f"Using {doc_source}()")
                print("First doc keys:", docs[0].keys())
                print("\nFirst doc:", docs[0])
                break
    except:
        continue
else:
    print("No docs found in any source")
