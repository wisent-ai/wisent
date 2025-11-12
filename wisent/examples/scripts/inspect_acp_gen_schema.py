#!/usr/bin/env python3
"""Inspect the schema of acp_prog_gen task."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from lm_eval.tasks import TaskManager

def main():
    tm = TaskManager()

    # Load acp_prog_gen task
    task = tm.load_task_or_group("acp_prog_gen")
    print(f"Task type: {type(task)}")

    # If it's a dict (group), get the first task
    if isinstance(task, dict):
        print(f"Is a group with {len(task)} subtasks:")
        for name in list(task.keys())[:5]:
            print(f"  - {name}")
        # Get first actual task
        first_task_name = list(task.keys())[0]
        task = task[first_task_name]
        print(f"\nUsing first task: {first_task_name}")

    print(f"Task name: {getattr(task, 'NAME', 'N/A')}")

    # Get a doc to see the schema
    docs = list(task.test_docs())[:2]
    if docs:
        print(f"\nNumber of test docs: {len(list(task.test_docs()))}")
        print(f"\nFirst doc keys: {list(docs[0].keys())}")
        print(f"\nFirst doc:")
        for k, v in docs[0].items():
            if isinstance(v, str) and len(v) > 200:
                print(f"  {k}: {v[:200]}...")
            else:
                print(f"  {k}: {v}")

        if len(docs) > 1:
            print(f"\nSecond doc:")
            for k, v in docs[1].items():
                if isinstance(v, str) and len(v) > 200:
                    print(f"  {k}: {v[:200]}...")
                else:
                    print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
