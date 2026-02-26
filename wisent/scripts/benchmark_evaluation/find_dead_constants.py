"""Find all dead constants across the entire codebase.

Extracts every UPPER_CASE constant from all definition files,
then greps each one to find constants with zero external consumers.
"""
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEFINITION_FILES = [
    "wisent/core/constants.py",
    "wisent/core/grp_05/sub_constants/validated/_validated.py",
    "wisent/core/grp_05/sub_constants/cannot_be_optimized/_fixed_01.py",
    "wisent/core/grp_05/sub_constants/cannot_be_optimized/_fixed_02.py",
    "wisent/core/grp_05/sub_constants/cannot_be_optimized/sub_fixed/_fixed_03.py",
    "wisent/core/grp_05/sub_constants/cannot_be_optimized/sub_fixed/_fixed_04.py",
    "wisent/core/grp_05/sub_constants/for_experiments/_exp_01.py",
    "wisent/core/grp_05/sub_constants/for_experiments/_exp_02.py",
    "wisent/core/grp_05/sub_constants/for_experiments/_exp_03.py",
    "wisent/core/grp_05/sub_constants/for_experiments/sub_exp/_exp_04.py",
    "wisent/core/grp_05/sub_constants/for_experiments/sub_exp/_exp_05.py",
    "wisent/core/grp_05/sub_constants/for_experiments/sub_exp/_exp_06.py",
    "wisent/core/grp_05/sub_constants/for_experiments/sub_exp/sub_exp2/_exp_07.py",
    "wisent/core/grp_05/sub_constants/for_experiments/sub_exp/sub_exp2/_exp_08.py",
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
    """Count files referencing this constant, excluding definition/classification files."""
    result = subprocess.run(
        ["grep", "-rl", "--include=*.py", name, REPO_ROOT],
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

    print(f"Total constants found: {len(all_constants)}")
    print("Checking each one for external references...\n")

    dead = []
    checked = 0
    for name, source_file in sorted(all_constants.items()):
        refs = count_external_refs(name)
        checked += 1
        if checked % 100 == 0:
            print(f"  ... checked {checked}/{len(all_constants)}", file=sys.stderr)
        if not refs:
            dead.append((name, source_file))

    print(f"\n{'='*60}")
    print(f"DEAD CONSTANTS (zero external consumers): {len(dead)}")
    print(f"{'='*60}")
    for name, source_file in dead:
        print(f"  {name}  ({source_file})")

    print(f"\nTotal checked: {checked}")
    print(f"Total dead: {len(dead)}")


if __name__ == "__main__":
    main()
