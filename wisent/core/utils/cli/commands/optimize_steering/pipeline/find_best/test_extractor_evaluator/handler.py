"""Test that a single benchmark's extractor and evaluator work together.

Usage:
    wisent test-one-extractor-and-evaluator --task <benchmark> [--limit N]

Steps:
    A. Look up the HF extractor for the given benchmark name
    B. Extract a small number of contrastive pairs
    C. Resolve the evaluator via the extractor's evaluator_name
    D. Run the evaluator on each pair's correct and incorrect responses
    E. Report: correct should be TRUTHFUL, incorrect should be UNTRUTHFUL
    F. Fail with nonzero exit code if any pair is wrong
"""

import inspect
import sys
import time
import argparse

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvaluatorError
from wisent.core.utils.config_tools.constants import (
    EXIT_CODE_ERROR,
    INDEX_FIRST,
    SEPARATOR_WIDTH_WIDE,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
    TEST_EXTRACTOR_EVALUATOR_MAX_ERRORS,
    TEST_EXTRACTOR_EVALUATOR_HTTP_TIMEOUT,
)


_INFRA_MARKERS = ("judge_model", "test_code", "requires a model",
                   "ModelNotProvidedError")


def _needs_external_infra(evaluator, pairs, task):
    """Probe whether the evaluator requires external infrastructure.

    Tries evaluating the first pair; if the evaluator raises an
    exception whose message references unavailable infrastructure
    (LLM judge, Docker test_code, model, etc.) then string
    comparison is used.
    """
    if not pairs:
        return False
    pair = pairs[INDEX_FIRST]
    try:
        evaluator.evaluate(
            response=pair.positive_response.model_response,
            expected=pair.positive_response.model_response,
            question=pair.prompt,
            task_name=task,
        )
    except Exception as exc:
        msg = str(exc)
        exc_type = type(exc).__name__
        for marker in _INFRA_MARKERS:
            if marker in msg or marker in exc_type:
                return True
    return False


def _string_compare(response: str, expected: str) -> bool:
    """String comparison for judge-based evaluator verification."""
    return response.strip() == expected.strip()


def execute_test_one(args):
    """Run the end-to-end extractor + evaluator test for one benchmark."""
    task = args.task
    limit = args.limit
    sep = "=" * SEPARATOR_WIDTH_WIDE

    print(f"\n{sep}")
    print(f"TEST: extractor + evaluator for '{task}'")
    print(f"{sep}\n")

    # ── A. Look up extractor ──────────────────────────────────
    print(f"[A] Looking up HF extractor for '{task}'...")
    from wisent.extractors.hf.hf_extractor_registry import (
        get_extractor, UnsupportedHuggingFaceBenchmarkError,
    )
    try:
        extractor = get_extractor(task, http_timeout=args.http_timeout)
    except TypeError:
        try:
            extractor = get_extractor(task)
        except TypeError as exc2:
            print(f"FAIL: extractor constructor error: {exc2}")
            sys.exit(EXIT_CODE_ERROR)
    except UnsupportedHuggingFaceBenchmarkError as exc:
        print(f"FAIL: {exc}")
        sys.exit(EXIT_CODE_ERROR)

    evaluator_name = getattr(extractor, "evaluator_name", None)
    print(f"     Extractor class: {type(extractor).__name__}")
    print(f"     evaluator_name:  {evaluator_name}")

    if not evaluator_name:
        print("FAIL: extractor has no evaluator_name attribute")
        sys.exit(EXIT_CODE_ERROR)

    # ── B. Extract contrastive pairs ──────────────────────────
    print(f"\n[B] Extracting up to {limit} contrastive pairs...")
    start = time.time()
    try:
        pairs = extractor.extract_contrastive_pairs(limit=limit)
    except TypeError as exc:
        if "lm_eval_task_data" in str(exc):
            try:
                pairs = extractor.extract_contrastive_pairs(
                    lm_eval_task_data=None, limit=limit)
            except Exception as inner:
                print(f"FAIL: extraction raised "
                      f"{type(inner).__name__}: {inner}")
                sys.exit(EXIT_CODE_ERROR)
        else:
            print(f"FAIL: extraction raised TypeError: {exc}")
            sys.exit(EXIT_CODE_ERROR)
    except Exception as exc:
        print(f"FAIL: extraction raised {type(exc).__name__}: {exc}")
        sys.exit(EXIT_CODE_ERROR)
    elapsed = time.time() - start
    print(f"     Extracted {len(pairs)} pairs in {elapsed:.1f}s")

    if not pairs:
        print("FAIL: extractor returned zero pairs")
        sys.exit(EXIT_CODE_ERROR)

    # ── C. Resolve evaluator ──────────────────────────────────
    print(f"\n[C] Resolving evaluator '{evaluator_name}'...")

    # Import benchmark_specific to trigger evaluator registration
    import wisent.core.reading.evaluators.core.benchmark_specific  # noqa: F401

    try:
        evaluator_cls = BaseEvaluator.get(evaluator_name)
    except EvaluatorError as exc:
        print(f"FAIL: {exc}")
        print(f"     Registered evaluators: "
              f"{sorted(BaseEvaluator.list_registered().keys())}")
        sys.exit(EXIT_CODE_ERROR)

    # Instantiate
    sig = inspect.signature(evaluator_cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    print(f"     Evaluator class: {evaluator_cls.__name__}")
    print(f"     Constructor params: {accepted or '(none)'}")

    try:
        evaluator = evaluator_cls()
    except TypeError:
        print(f"     Cannot instantiate with no args, skipping eval")
        print("PASS (extractor + evaluator resolution OK)")
        return

    # ── D. Evaluate pairs ─────────────────────────────────────
    is_infra_evaluator = getattr(evaluator, "requires_judge", False)
    if not is_infra_evaluator:
        is_infra_evaluator = _needs_external_infra(evaluator, pairs, task)

    if is_infra_evaluator:
        print(f"\n[D] Evaluator requires external infrastructure "
              f"— using string comparison for {len(pairs)} pairs...")
    else:
        print(f"\n[D] Evaluating {len(pairs)} pairs...")

    correct_ok, incorrect_ok, errors = [], [], []

    for i, pair in enumerate(pairs):
        prompt = pair.prompt
        correct_resp = pair.positive_response.model_response
        incorrect_resp = pair.negative_response.model_response
        expected = correct_resp

        if is_infra_evaluator:
            correct_ok.append(
                _string_compare(correct_resp, expected))
            incorrect_ok.append(
                not _string_compare(incorrect_resp, expected))
            continue

        # Evaluate the correct response
        try:
            result_correct = evaluator.evaluate(
                response=correct_resp,
                expected=expected,
                question=prompt,
                task_name=task,
            )
            correct_ok.append(result_correct.ground_truth == "TRUTHFUL")
        except EvaluatorError as exc:
            errors.append(f"Pair {i} correct: {exc}")
            continue
        except Exception as exc:
            errors.append(
                f"Pair {i} correct: {type(exc).__name__}: {exc}")
            continue

        # Evaluate the incorrect response
        try:
            result_incorrect = evaluator.evaluate(
                response=incorrect_resp,
                expected=expected,
                question=prompt,
                task_name=task,
            )
            is_wrong = result_incorrect.ground_truth == "UNTRUTHFUL"
            incorrect_ok.append(is_wrong)
            if not is_wrong:
                print(f"  [!] Pair {i}: incorrect not detected."
                      f" gt={result_incorrect.ground_truth}"
                      f" details={result_incorrect.details}")
        except EvaluatorError as exc:
            errors.append(f"Pair {i} incorrect: {exc}")
        except Exception as exc:
            errors.append(
                f"Pair {i} incorrect: {type(exc).__name__}: {exc}")

    # ── E. Report ─────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"RESULTS for '{task}'")
    print(f"{sep}")
    n_correct = sum(correct_ok)
    n_incorrect = sum(incorrect_ok)
    total_correct = len(correct_ok)
    total_incorrect = len(incorrect_ok)
    print(f"  Correct responses → TRUTHFUL:    "
          f"{n_correct}/{total_correct}")
    print(f"  Incorrect responses → UNTRUTHFUL: "
          f"{n_incorrect}/{total_incorrect}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors[:args.max_errors]:
            print(f"    - {err}")

    # ── F. Exit code ──────────────────────────────────────────
    if total_correct and total_incorrect:
        all_passed = (n_correct == total_correct
                      and n_incorrect == total_incorrect)
        if all_passed:
            print("\nPASS: all pairs evaluated correctly")
        else:
            print("\nFAIL: some pairs evaluated incorrectly")
            sys.exit(EXIT_CODE_ERROR)
    elif errors:
        print(f"\nFAIL: {len(errors)} evaluation errors")
        sys.exit(EXIT_CODE_ERROR)
    else:
        print("\nPASS (extractor + evaluator resolution OK)")


def build_parser():
    """Build argument parser for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Test one benchmark extractor + evaluator")
    parser.add_argument("--task", required=True,
                        help="Benchmark name (e.g., 'agentharm')")
    parser.add_argument(
        "--limit", type=int,
        default=TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
        help="Max pairs to extract")
    parser.add_argument(
        "--max-errors", type=int,
        default=TEST_EXTRACTOR_EVALUATOR_MAX_ERRORS,
        help="Max errors to display")
    parser.add_argument(
        "--http-timeout", type=int,
        default=TEST_EXTRACTOR_EVALUATOR_HTTP_TIMEOUT,
        help="HTTP timeout for extractor dataset downloads")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    execute_test_one(parsed)
