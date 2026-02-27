"""Analyze all constants across the entire codebase.

Extracts every UPPER_CASE constant from all definition files,
counts external consumers for each, and reports usage statistics.
"""
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEFINITION_FILES = [
    "wisent/core/constants.py",
    "wisent/core/infrastructure/constant_definitions/validated/_validated.py",
    "wisent/core/infrastructure/constant_definitions/cannot_be_optimized/_display_viz.py",
    "wisent/core/infrastructure/constant_definitions/cannot_be_optimized/_infrastructure.py",
    "wisent/core/infrastructure/constant_definitions/cannot_be_optimized/_benchmark_data.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_domain/_optimization.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_domain/_evaluation.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_domain/analysis/_analysis.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_domain/analysis/_geometry.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_domain/analysis/infra/_agent_marketplace.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_domain/analysis/infra/_data_infra.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/_grom.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/_tecza.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/_tetno.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/simple/_broyden.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/simple/_caa.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/simple/_mlp.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/simple/_ostrze.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/transport/_nurt.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/transport/_przelom.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/transport/_szlak.py",
    "wisent/core/infrastructure/constant_definitions/for_experiments/by_method/transport/_wicher.py",
]

SKIP_FILES = set()
for f in DEFINITION_FILES:
    SKIP_FILES.add(os.path.basename(f))
SKIP_FILES.add("reorg_classify.py")
SKIP_FILES.add("reorg_main.py")
SKIP_FILES.add("find_dead_constants.py")
SKIP_FILES.add("__init__.py")

CONSTANT_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=", re.MULTILINE)


def extract_constants(filepath):
    """Extract all UPPER_CASE = ... assignments from a file."""
    with open(filepath) as f:
        content = f.read()
    return CONSTANT_RE.findall(content)


def count_external_refs(name):
    """Count files referencing this constant (word-boundary match)."""
    result = subprocess.run(
        ["grep", "-rlw", "--include=*.py", name, REPO_ROOT],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    files = result.stdout.strip().split("\n")
    external = []
    for f in files:
        if not f:
            continue
        basename = os.path.basename(f)
        if basename in SKIP_FILES:
            continue
        external.append(f)
    return external


def main():
    all_constants = {}
    for rel_path in DEFINITION_FILES:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(full_path):
            continue
        names = extract_constants(full_path)
        for name in names:
            if name not in all_constants:
                all_constants[name] = rel_path

    total = len(all_constants)
    print(f"Total constants found: {total}")

    usage = {}
    checked = 0
    for name, source_file in sorted(all_constants.items()):
        refs = count_external_refs(name)
        usage[name] = (source_file, len(refs), refs)
        checked += 1
        if checked % 100 == 0:
            print(f"  ... checked {checked}/{total}", file=sys.stderr)

    dead = [(n, s, c, r) for n, (s, c, r) in usage.items() if c == 0]
    single = [(n, s, c, r) for n, (s, c, r) in usage.items() if c == 1]
    few = [(n, s, c, r) for n, (s, c, r) in usage.items() if c == 2]
    multi = [(n, s, c, r) for n, (s, c, r) in usage.items() if c >= 3]

    print(f"\n{'='*70}")
    print(f"USAGE SUMMARY")
    print(f"{'='*70}")
    print(f"Dead (0 uses):      {len(dead)}")
    print(f"Single-use (1 use): {len(single)}")
    print(f"Two uses:           {len(few)}")
    print(f"Multi-use (3+):     {len(multi)}")
    print(f"Total:              {total}")

    print(f"\n{'='*70}")
    print(f"DEAD CONSTANTS (0 external consumers)")
    print(f"{'='*70}")
    for name, source, count, refs in sorted(dead):
        print(f"  {name}  ({source})")

    print(f"\n{'='*70}")
    print(f"SINGLE-USE CONSTANTS (1 external consumer)")
    print(f"{'='*70}")
    for name, source, count, refs in sorted(single):
        consumer = refs[0].replace(REPO_ROOT + "/", "")
        print(f"  {name}  ->  {consumer}")

    print(f"\n{'='*70}")
    print(f"TWO-USE CONSTANTS (2 external consumers)")
    print(f"{'='*70}")
    for name, source, count, refs in sorted(few):
        consumers = [r.replace(REPO_ROOT + "/", "") for r in refs]
        print(f"  {name}  ->  {', '.join(consumers)}")


if __name__ == "__main__":
    main()
