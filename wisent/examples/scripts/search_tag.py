#!/usr/bin/env python3
"""Search for tasks containing 'tag' in lm-eval."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from lm_eval.tasks import TaskManager

def main():
    tm = TaskManager()

    # Search for any task containing "tag" (case-insensitive)
    tag_tasks = [t for t in tm.task_index.keys() if 'tag' in t.lower()]
    print(f"Found {len(tag_tasks)} tasks containing 'tag':")
    for task in sorted(tag_tasks):
        print(f"  - {task}")

    # Search for 3-4 letter tasks starting with 't'
    short_t_tasks = [t for t in tm.task_index.keys() if t.lower().startswith('t') and len(t) <= 4]
    print(f"\nFound {len(short_t_tasks)} short tasks (3-4 letters) starting with 't':")
    for task in sorted(short_t_tasks):
        print(f"  - {task}")

    # Check some specific alternatives
    alternatives = ["tqa", "tmqa", "tag", "TAG", "taq", "tga", "atag"]
    print(f"\nChecking specific alternatives:")
    for alt in alternatives:
        if alt in tm.task_index:
            print(f"  ✓ Found: {alt}")
        else:
            print(f"  ✗ Not found: {alt}")

if __name__ == "__main__":
    main()
