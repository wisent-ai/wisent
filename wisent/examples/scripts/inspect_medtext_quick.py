from lm_eval.tasks import TaskManager
import sys

task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("medtext")
task_obj = list(task_dict.values())[0]

# Try different doc sources
for doc_source in ['test_docs', 'validation_docs', 'train_docs']:
    try:
        docs_fn = getattr(task_obj, doc_source, None)
        if docs_fn:
            print(f'Trying {doc_source}...')
            docs_iter = docs_fn()
            # Get just first doc
            first_doc = next(docs_iter)
            print(f'Using {doc_source}')
            print(f'Keys: {list(first_doc.keys())}')
            print(f'\nFirst doc: {first_doc}')
            sys.exit(0)
    except Exception as e:
        print(f'Failed {doc_source}: {e}')
        continue

print("Could not get any docs")
