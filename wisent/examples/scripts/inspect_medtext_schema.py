from lm_eval.tasks import TaskManager

task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("medtext")
task_obj = list(task_dict.values())[0]

# Try different doc sources
for doc_source in ['validation_docs', 'test_docs', 'train_docs']:
    try:
        docs_fn = getattr(task_obj, doc_source, None)
        if docs_fn:
            docs = list(docs_fn())
            if docs:
                print(f'Using {doc_source}')
                print(f'Keys: {list(docs[0].keys())}')
                print(f'\nFirst doc: {docs[0]}')
                break
    except:
        continue
