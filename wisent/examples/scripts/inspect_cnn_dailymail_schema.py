from lm_eval import tasks
from lm_eval.tasks import TaskManager

task_manager = TaskManager()
task_dict = task_manager.load_task_or_group("cnn_dailymail")
task_obj = list(task_dict.values())[0]

# Load one doc
docs = list(task_obj.validation_docs())
if docs:
    print("First doc keys:", docs[0].keys())
    print("First doc sample:", {k: str(v)[:200] if isinstance(v, str) and len(str(v)) > 200 else v for k, v in docs[0].items()})
else:
    print("No docs found")
