"""Resolve a constant X using the heuristic rules.

Rules from remaining_constants_heuristics.md, applied top-to-bottom:
    Rule 1: Zero references outside definition → DELETE
    Rule 2: Hardware-dependent → RUNTIME FUNCTION
    Rule 3: Model-dependent → DERIVE FROM MODEL
    Rule 4: Definition, not a choice → KEEP
    Rule 5: Nothing matched → KEEP
    Rule 6: Half/Double test — needs justification or experiment

Usage:
    python resolve_constant.py GROM_LEARNING_RATE
    python resolve_constant.py --all
"""
import argparse
import os
import re
import subprocess
import sys

from _resolve_patterns import (
    classify_rule2, classify_rule3, classify_rule4,
    half_double_trivially_passes,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

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

CONSTANT_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=\s*(.+)", re.MULTILINE)

SKIP_BASENAMES = {
    "__init__.py", "resolve_constant.py", "find_dead_constants.py",
    "reorg_classify.py", "reorg_main.py", "_resolve_patterns.py",
}
for _f in DEFINITION_FILES:
    SKIP_BASENAMES.add(os.path.basename(_f))


def extract_constants_with_values(filepath):
    """Return list of (name, raw_value, line_num, has_comment, filepath)."""
    results = []
    with open(filepath) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        m = CONSTANT_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        raw_val = m.group(2).strip()
        has_comment = "#" in raw_val
        if has_comment:
            raw_val = raw_val[:raw_val.index("#")].strip()
        # Check preceding line for section comment
        if not has_comment and i >= 2:
            prev = lines[i - 2].strip()
            if prev.startswith("#"):
                has_comment = True
        results.append((name, raw_val, i, has_comment, filepath))
    return results


def count_external_refs(name):
    """Count .py files referencing this constant outside definition files."""
    result = subprocess.run(
        ["grep", "-rlw", "--include=*.py", name, REPO_ROOT],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    external = []
    for f in result.stdout.strip().split("\n"):
        if not f:
            continue
        if os.path.basename(f) in SKIP_BASENAMES:
            continue
        external.append(f.replace(REPO_ROOT + "/", ""))
    return external


def resolve_one(name, all_constants):
    """Apply heuristic rules to constant. Returns a result dict."""
    matches = [c for c in all_constants if c[0] == name]
    if not matches:
        return {"name": name, "rule": "NOT_FOUND",
                "action": "SKIP",
                "reason": f"Not found in any definition file"}

    cname, raw_val, line_num, has_comment, source_file = matches[0]
    rel_source = source_file.replace(REPO_ROOT + "/", "")
    refs = count_external_refs(name)
    ref_count = len(refs)
    base = {"name": name, "source": rel_source,
            "line": line_num, "value": raw_val, "refs": ref_count}

    if ref_count == 0:
        return {**base, "rule": "RULE_1_DEAD", "action": "DELETE",
                "reason": "Zero external references"}

    if classify_rule2(name):
        return {**base, "rule": "RULE_2_HARDWARE",
                "action": "RUNTIME_FUNCTION",
                "reason": "Value should depend on hardware",
                "consumers": refs[:5]}

    if classify_rule3(name):
        return {**base, "rule": "RULE_3_MODEL",
                "action": "DERIVE_FROM_MODEL",
                "reason": "Value should depend on model architecture",
                "consumers": refs[:5]}

    if classify_rule4(name):
        return {**base, "rule": "RULE_4_DEFINITION", "action": "KEEP",
                "reason": "Changing this changes the concept, not quality",
                "has_justification": has_comment}

    # Rule 5 + Rule 6
    if half_double_trivially_passes(raw_val):
        j_status = "N/A (non-numeric)"
    elif has_comment:
        j_status = "HAS_COMMENT"
    else:
        j_status = "NEEDS_JUSTIFICATION"

    return {**base, "rule": "RULE_5_DEFAULT_KEEP", "action": "KEEP",
            "reason": f"Half/double test: {j_status}",
            "has_justification": has_comment,
            "justification_status": j_status}


def load_all_constants():
    """Load all constants from all definition files."""
    all_constants = []
    for rel_path in DEFINITION_FILES:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(full_path):
            continue
        all_constants.extend(extract_constants_with_values(full_path))
    return all_constants


def print_result(result):
    """Print a single resolution result."""
    sep = "=" * 70
    print(sep)
    print(f"  Constant: {result['name']}")
    if "source" in result:
        print(f"  Defined:  {result['source']}:{result.get('line', '?')}")
    if "value" in result:
        print(f"  Value:    {result['value']}")
    if "refs" in result:
        print(f"  Refs:     {result['refs']} external consumers")
    print(f"  Rule:     {result['rule']}")
    print(f"  Action:   {result['action']}")
    print(f"  Reason:   {result['reason']}")
    if "justification_status" in result:
        print(f"  Justif:   {result['justification_status']}")
    if "consumers" in result:
        for c in result["consumers"]:
            print(f"            -> {c}")
    print(sep)


def print_summary(all_constants):
    """Classify all constants and print report."""
    counts = {
        "RULE_1_DEAD": [], "RULE_2_HARDWARE": [],
        "RULE_3_MODEL": [], "RULE_4_DEFINITION": [],
        "RULE_5_DEFAULT_KEEP": [],
    }
    needs_justification = []
    total = len(all_constants)
    done = 0

    for cname, raw_val, line_num, has_comment, source_file in all_constants:
        result = resolve_one(cname, all_constants)
        rule = result["rule"]
        if rule in counts:
            counts[rule].append(result)
        if result.get("justification_status") == "NEEDS_JUSTIFICATION":
            needs_justification.append(result)
        done += 1
        if done % 50 == 0:
            print(f"  ... classified {done}/{total}", file=sys.stderr)

    sep = "=" * 70
    print(sep)
    print("CONSTANT CLASSIFICATION SUMMARY")
    print(sep)
    print(f"  Total constants:           {total}")
    print(f"  Rule 1 (dead, DELETE):     {len(counts['RULE_1_DEAD'])}")
    print(f"  Rule 2 (hardware, FUNC):   {len(counts['RULE_2_HARDWARE'])}")
    print(f"  Rule 3 (model, DERIVE):    {len(counts['RULE_3_MODEL'])}")
    print(f"  Rule 4 (definition, KEEP): {len(counts['RULE_4_DEFINITION'])}")
    print(f"  Rule 5 (default, KEEP):    {len(counts['RULE_5_DEFAULT_KEEP'])}")
    print(f"  Needs justification:       {len(needs_justification)}")
    print(sep)

    for label, key in [
        ("DEAD (delete these)", "RULE_1_DEAD"),
        ("HARDWARE-DEPENDENT (runtime function)", "RULE_2_HARDWARE"),
        ("MODEL-DEPENDENT (derive from model)", "RULE_3_MODEL"),
    ]:
        if counts[key]:
            print(f"\n--- {label} ---")
            for r in sorted(counts[key], key=lambda x: x["name"]):
                print(f"  {r['name']} = {r['value']}  ({r['source']})")

    if needs_justification:
        print(f"\n--- NEEDS JUSTIFICATION (half/double test) ---")
        for r in sorted(needs_justification, key=lambda x: x["name"]):
            print(f"  {r['name']} = {r['value']}  ({r['source']})")


def main():
    parser = argparse.ArgumentParser(
        description="Resolve a constant using heuristic rules"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "constant", nargs="?", default=None,
        help="Name of constant (e.g. GROM_LEARNING_RATE)")
    group.add_argument("--all", action="store_true",
                       help="Classify all constants")
    args = parser.parse_args()

    all_constants = load_all_constants()
    print(f"Loaded {len(all_constants)} constants", file=sys.stderr)

    if args.constant:
        result = resolve_one(args.constant, all_constants)
        print_result(result)
    elif args.all:
        print_summary(all_constants)


if __name__ == "__main__":
    main()
