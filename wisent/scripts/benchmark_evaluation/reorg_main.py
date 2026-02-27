"""Main script for reorganizing arbitrary constants into two chains."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from reorg_classify import is_cannot_optimize

BASE = "wisent/core/infrastructure/constant_definitions"
ARB_FILES = [
    f"{BASE}/arbitrary/_arb_01.py",
    f"{BASE}/arbitrary/_arb_02.py",
    f"{BASE}/arbitrary/_arb_03.py",
    f"{BASE}/arbitrary/_arb_04.py",
    f"{BASE}/arbitrary/sub_arb/_arb_05.py",
    f"{BASE}/arbitrary/sub_arb/_arb_06.py",
    f"{BASE}/arbitrary/sub_arb/_arb_07.py",
    f"{BASE}/arbitrary/sub_arb/_arb_08.py",
    f"{BASE}/arbitrary/sub_arb/sub_arb2/_arb_09.py",
    f"{BASE}/arbitrary/sub_arb/sub_arb2/_arb_10.py",
    f"{BASE}/arbitrary/sub_arb/sub_arb2/_arb_11.py",
    f"{BASE}/arbitrary/sub_arb/sub_arb2/_arb_12.py",
    f"{BASE}/arbitrary/sub_arb/sub_arb2/sub_arb3/_inference.py",
]

# NOTE: EXP_FILES is stale. for_experiments/ now uses by_domain/ + by_method/ chains.
# See find_dead_constants.py for current file list.
EXP_FILES = []

FIXED_FILES = [
    (f"{BASE}/cannot_be_optimized/_display_viz.py", f"{BASE}/cannot_be_optimized/_infrastructure.py"),
    (f"{BASE}/cannot_be_optimized/_infrastructure.py", f"{BASE}/cannot_be_optimized/_benchmark_data.py"),
    (f"{BASE}/cannot_be_optimized/_benchmark_data.py", None),
]


def parse_arb_file(filepath):
    """Parse an arb file, return list of entries with comments and assignment."""
    entries = []
    pending_comments = []
    with open(filepath) as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith('"""') or stripped.startswith("from "):
                continue
            if stripped == "":
                pending_comments.append("")
                continue
            if stripped.startswith("#"):
                pending_comments.append(stripped)
                continue
            match = re.match(r"^([A-Z][A-Z0-9_]*)\s*=", stripped)
            if match:
                entries.append({"comments": list(pending_comments), "name": match.group(1), "line": stripped})
                pending_comments = []
            elif entries and (stripped.startswith("    ") or stripped.startswith(")")):
                entries[-1]["line"] += "\n" + stripped
    return entries


def entry_lines(entry):
    """Count how many lines an entry takes."""
    return len(entry["comments"]) + entry["line"].count("\n") + 1


def make_import_line(next_target):
    """Create import line for next file in chain."""
    if next_target is None:
        return None
    mod = next_target.replace("/", ".").replace(".py", "")
    return f"from {mod} import *  # noqa: F401,F403"


def write_chain_file(filepath, entries, next_target, docstring):
    """Write a single chain file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = [f'"""{docstring}"""']
    imp = make_import_line(next_target)
    if imp:
        lines.append(imp)
    lines.append("")
    for entry in entries:
        for c in entry["comments"]:
            lines.append(c)
        lines.append(entry["line"])
    lines.append("")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"  {filepath}: {len(lines)} lines, {len(entries)} constants")


def write_init(dirpath, first_file, docstring):
    """Write __init__.py for a chain directory."""
    mod = first_file.replace("/", ".").replace(".py", "")
    content = f'"""{docstring}"""\nfrom {mod} import *  # noqa: F401,F403\n'
    with open(f"{dirpath}/__init__.py", "w") as f:
        f.write(content)
    print(f"  {dirpath}/__init__.py")


def distribute(entries, file_specs, docstring, max_lines=278):
    """Distribute entries across file specs."""
    chunks = []
    current = []
    current_count = 5
    for entry in entries:
        el = entry_lines(entry)
        if current_count + el > max_lines and current:
            chunks.append(current)
            current = []
            current_count = 5
        current.append(entry)
        current_count += el
    if current:
        chunks.append(current)
    if len(chunks) > len(file_specs):
        print(f"WARNING: {len(chunks)} chunks but only {len(file_specs)} file slots")
        while len(chunks) > len(file_specs):
            chunks[-2].extend(chunks[-1])
            chunks.pop()
    while len(chunks) < len(file_specs):
        chunks.append([])
    for i, (filepath, next_target) in enumerate(file_specs):
        write_chain_file(filepath, chunks[i], next_target, docstring)


def main():
    exp_entries = []
    fixed_entries = []
    for filepath in ARB_FILES:
        for entry in parse_arb_file(filepath):
            if is_cannot_optimize(entry["name"]):
                fixed_entries.append(entry)
            else:
                exp_entries.append(entry)
    print(f"For experiments: {len(exp_entries)} constants")
    print(f"Cannot be optimized: {len(fixed_entries)} constants")
    exp_doc = "Constants requiring experimental optimization."
    print(f"\n--- for_experiments chain ---")
    if EXP_FILES:
        distribute(exp_entries, EXP_FILES, exp_doc)
        write_init(f"{BASE}/for_experiments", EXP_FILES[0][0], exp_doc)
    fixed_doc = "Constants that cannot be experimentally optimized."
    print(f"\n--- cannot_be_optimized chain ---")
    distribute(fixed_entries, FIXED_FILES, fixed_doc)
    write_init(f"{BASE}/cannot_be_optimized", FIXED_FILES[0][0], fixed_doc)
    print("\nDone! Update constants.py to import from both new chains.")


if __name__ == "__main__":
    main()
