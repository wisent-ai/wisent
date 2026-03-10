#!/usr/bin/env python3
"""Compute semantic version bump level from __init__.py changes and git diff.

Pure-stdlib script. No wisent imports.

Args:
    --old-init <path>      Path to the old wisent/__init__.py (from git tag)
    --current-init <path>  Path to the current wisent/__init__.py
    --diff-summary <path>  Path to git diff --name-status output

Output:
    Prints one of: "major", "minor", "patch" to stdout.
"""

import argparse
import ast
import sys

from constants import (
    ADDED_INCREMENT,
    EXPECTED_DIFF_PARTS,
    SPLIT_MAXSPLIT,
    ZERO_COUNT,
)


def extract_all_names(filepath: str) -> set[str]:
    """Extract names from __all__ in a Python file using ast."""
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id != "__all__":
                continue
            if not isinstance(node.value, ast.List):
                continue
            names = set()
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(
                    elt.value, str
                ):
                    names.add(elt.value)
            return names
    return set()


def parse_diff_summary(filepath: str) -> list[tuple[str, str]]:
    """Parse git diff --name-status output into (status, path) tuples."""
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", SPLIT_MAXSPLIT)
            if len(parts) == EXPECTED_DIFF_PARTS:
                entries.append((parts[ZERO_COUNT], parts[SPLIT_MAXSPLIT]))
    return entries


def count_added_py_files(entries: list[tuple[str, str]]) -> int:
    """Count newly added .py files (excluding __init__.py)."""
    count = ZERO_COUNT
    for status, path in entries:
        if not status.startswith("A"):
            continue
        if not path.endswith(".py"):
            continue
        basename = path.rsplit("/", SPLIT_MAXSPLIT)[-SPLIT_MAXSPLIT]
        if basename == "__init__.py":
            continue
        count += ADDED_INCREMENT
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute semantic version bump level."
    )
    parser.add_argument(
        "--old-init", required=True, help="Path to old __init__.py"
    )
    parser.add_argument(
        "--current-init", required=True, help="Path to current __init__.py"
    )
    parser.add_argument(
        "--diff-summary", required=True, help="Path to diff --name-status"
    )
    args = parser.parse_args()

    old_names = extract_all_names(args.old_init)
    current_names = extract_all_names(args.current_init)

    removed = old_names - current_names
    if removed:
        print("major")
        return

    diff_entries = parse_diff_summary(args.diff_summary)
    added_count = count_added_py_files(diff_entries)
    if added_count > ZERO_COUNT:
        print("minor")
        return

    print("patch")


if __name__ == "__main__":
    main()
