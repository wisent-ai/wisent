"""Fix the order of correct/incorrect answers in extractor files.

The correct order is:
    A. {incorrect}
    B. {correct}

This script:
1. Finds files with the reversed order and fixes them
2. Checks if evaluator_name == "log_likelihoods" and verifies A/B pattern is present
"""

import re
from pathlib import Path


def fix_extractor_order():
    """Find and fix extractors with incorrect A/B order."""

    # Directories to search
    base_path = Path(__file__).parent.parent / "core" / "contrastive_pairs"
    search_dirs = [
        base_path / "lm_eval_pairs" / "lm_task_extractors",
        base_path / "huggingface_pairs" / "hf_task_extractors",
    ]

    # Pattern for incorrect order (correct first, incorrect second)
    incorrect_pattern = r'\\nA\. \{correct\}\\nB\. \{incorrect\}'

    # What it should be replaced with
    correct_replacement = r'\\nA. {incorrect}\\nB. {correct}'

    # Pattern for correct order
    correct_pattern = r'\\nA\. \{incorrect\}\\nB\. \{correct\}'

    # Pattern for log_likelihoods evaluator
    log_likelihood_pattern = r'evaluator_name\s*=\s*["\']log_likelihood[s]?["\']'

    files_with_incorrect_order = []
    log_likelihood_missing_ab = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"Directory not found: {search_dir}")
            continue

        for py_file in search_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            content = py_file.read_text()

            # Check if file has incorrect order
            if re.search(incorrect_pattern, content):
                files_with_incorrect_order.append(py_file)

                # Fix the order
                fixed_content = re.sub(
                    incorrect_pattern,
                    correct_replacement,
                    content
                )

                py_file.write_text(fixed_content)

            # Check if evaluator is log_likelihoods but missing A/B pattern
            has_log_likelihood = re.search(log_likelihood_pattern, content)
            has_ab_pattern = re.search(correct_pattern, content) or re.search(incorrect_pattern, content)

            if has_log_likelihood and not has_ab_pattern:
                log_likelihood_missing_ab.append(py_file)

    # Report results
    print("=" * 60)
    print("EXTRACTOR ORDER FIX REPORT")
    print("=" * 60)

    print(f"\n1. Files with incorrect order (A.correct/B.incorrect -> fixed): {len(files_with_incorrect_order)}")
    if files_with_incorrect_order:
        print("\n   Fixed files:")
        for f in sorted(files_with_incorrect_order):
            print(f"     - {f.name}")

    print(f"\n2. Files with log_likelihoods evaluator but MISSING A/B pattern: {len(log_likelihood_missing_ab)}")
    if log_likelihood_missing_ab:
        print("\n   Missing A/B pattern:")
        for f in sorted(log_likelihood_missing_ab):
            print(f"     - {f.name}")

    return {
        "fixed": files_with_incorrect_order,
        "missing_ab": log_likelihood_missing_ab,
    }


if __name__ == "__main__":
    fix_extractor_order()
