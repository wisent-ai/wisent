#!/usr/bin/env python3
"""Inspect teca task schema."""

from lm_eval.tasks import TaskManager

def main():
    # Load task manager
    tm = TaskManager()

    # Check what "teca" is
    if "teca" in tm.task_index:
        print("✓ Found teca in task index")
        print(f"Task info: {tm.task_index['teca']}")

        # Try to load the task
        task = tm.load_task_or_group("teca")
        print(f"\nTask type: {type(task)}")
        print(f"Task name: {getattr(task, 'NAME', 'N/A')}")

        # Check if it's a group
        if hasattr(task, 'group'):
            print(f"Group: {task.group}")

        # Load some docs to see schema
        docs = task.test_docs() if hasattr(task, 'test_docs') else []
        docs_list = list(docs)[:1]
        if docs_list:
            print(f"\nSample doc keys: {list(docs_list[0].keys())}")
            print(f"Sample doc:")
            for k, v in docs_list[0].items():
                print(f"  {k}: {v}")
    else:
        print("✗ teca not found")

if __name__ == "__main__":
    main()
