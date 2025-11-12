from lm_eval import tasks
from lm_eval.tasks import TaskManager

task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("dbpedia_14")
task_obj = list(task_dict.values())[0]

# Load one doc
docs = list(task_obj.test_docs())
if docs:
    print("First doc keys:", docs[0].keys())
    print("\nFirst doc:", docs[0])
else:
    print("No docs found")
