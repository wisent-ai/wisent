#!/usr/bin/env python3
"""Check if acp_bench_hard exists in lm-eval."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from lm_eval.tasks import TaskManager

def main():
    tm = TaskManager()

    # Check for acp_bench_hard and related
    acp_tasks = [t for t in tm.task_index.keys() if 'acp' in t.lower() and 'bench' in t.lower()]
    print(f"Found {len(acp_tasks)} tasks with 'acp' and 'bench':")
    for task in sorted(acp_tasks):
        print(f"  - {task}")

    # Check specific names
    check_names = ["acp_bench_hard", "acp_bench", "acpbench", "acp_hard"]
    print(f"\nChecking specific names:")
    for name in check_names:
        if name in tm.task_index:
            print(f"  ✓ Found: {name}")
        else:
            print(f"  ✗ Not found: {name}")

if __name__ == "__main__":
    main()
