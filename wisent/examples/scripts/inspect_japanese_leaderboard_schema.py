from lm_eval import tasks
from lm_eval.tasks import TaskManager

task_manager = TaskManager()

# Check if japanese_leaderboard is a group or task
try:
    task_dict = task_manager.load_task_or_group("japanese_leaderboard")
    print(f"Loaded japanese_leaderboard: {type(task_dict)}")
    print(f"Number of items: {len(task_dict)}")

    # List first few keys
    keys = list(task_dict.keys())[:10]
    print(f"\nFirst 10 keys: {keys}")
except Exception as e:
    print(f"Error: {e}")
