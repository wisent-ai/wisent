#!/usr/bin/env python3
"""Search for all short task names that might match Tag."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from lm_eval.tasks import TaskManager

def main():
    tm = TaskManager()

    # Get all 3-letter task names
    three_letter = [t for t in tm.task_index.keys() if len(t) == 3]
    print(f"Found {len(three_letter)} tasks with exactly 3 letters:")
    for task in sorted(three_letter):
        print(f"  - {task}")

    # Get all 3-4 letter task names starting with T
    short_t = [t for t in tm.task_index.keys() if t.lower().startswith('t') and 3 <= len(t) <= 4]
    print(f"\nFound {len(short_t)} tasks with 3-4 letters starting with 't':")
    for task in sorted(short_t):
        print(f"  - {task}")

    # Search for anything with T, A, G in sequence (case insensitive)
    tag_pattern = [t for t in tm.task_index.keys() if 't' in t.lower() and 'a' in t.lower() and 'g' in t.lower()]
    print(f"\nFound {len(tag_pattern)} tasks containing t, a, and g:")
    for task in sorted(tag_pattern)[:20]:  # Show first 20
        print(f"  - {task}")

if __name__ == "__main__":
    main()
