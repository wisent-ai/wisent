"""Extracted CSV/JSON loading test helpers and additional tests."""

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from wisent.cli_utils.cli_prepare_dataset import (
    _load_from_csv_json,
    PrepState,
    Caps,
)
from wisent.core.utils.config_tools.constants import DEFAULT_SPLIT_RATIO, JSON_INDENT, SEPARATOR_WIDTH_REPORT


def create_test_csv(path: Path, data: List[Dict[str, str]]):
    """Create a test CSV file."""
    if not data:
        path.write_text("")
        return

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def create_test_json(path: Path, data: List[Dict[str, Any]]):
    """Create a test JSON file."""
    path.write_text(json.dumps(data, indent=JSON_INDENT))


def test_deterministic_split():
    """Test 7: Verify deterministic splitting with seed"""

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"

        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(20)
        ]
        create_test_csv(csv_path, test_data)

        # Run twice with same seed - using actual ContrastivePairSet
        result1 = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=DEFAULT_SPLIT_RATIO,
            seed=42,
            verbose=False,
        )

        result2 = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=DEFAULT_SPLIT_RATIO,
            seed=42,
            verbose=False,
        )

        # Check if splits are identical
        assert result1.qa_pairs == result2.qa_pairs, "Same seed should produce identical train splits"
        assert result1.test_source == result2.test_source, "Same seed should produce identical test splits"

        # Run with different seed
        result3 = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=DEFAULT_SPLIT_RATIO,
            seed=99,
            verbose=False,
        )

        # Should be different
        assert result1.qa_pairs != result3.qa_pairs, "Different seeds should produce different splits"


def test_different_column_names():
    """Test 8: CSV with custom column names"""

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "custom.csv"

        # Use different column names
        test_data = [
            {"query": "What is 2+2?", "right": "4", "wrong": "5"},
            {"query": "What is 3+3?", "right": "6", "wrong": "7"},
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["query", "right", "wrong"])
            writer.writeheader()
            writer.writerows(test_data)

        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="query",
            correct_col="right",
            incorrect_col="wrong",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=DEFAULT_SPLIT_RATIO,
            seed=42,
            verbose=False,
        )

        # Just verify that custom columns work and we get results
        assert result is not None, "Result is None with custom columns"
        assert len(result.qa_pairs) + len(result.test_source) == 2, "Custom columns should load all data"


def test_split_ratios():
    """Test 9: Various split ratios"""

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"

        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(100)
        ]
        create_test_csv(csv_path, test_data)

        test_ratios = [0.5, 0.7, 0.9, 0.99]

        for ratio in test_ratios:
            # Use actual ContrastivePairSet - no mocking
            result = _load_from_csv_json(
                from_csv=True,
                from_json=False,
                task_name=str(csv_path),
                question_col="question",
                correct_col="correct_answer",
                incorrect_col="incorrect_answer",
                limit=None,
                caps=Caps(train=1000, test=1000),  # High caps to test ratio
                split_ratio=ratio,
                seed=42,
                verbose=False,
            )

            expected_train = int(100 * ratio)

            # Allow +/-1 for rounding
            assert abs(len(result.qa_pairs) - expected_train) <= 1, \
                f"Ratio {ratio}: Expected ~{expected_train} train, got {len(result.qa_pairs)}"


def run_all_tests(test_functions):
    """Run all CSV/JSON loading tests."""
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("COMPREHENSIVE TESTING OF _load_from_csv_json() FUNCTION")
    print("=" * SEPARATOR_WIDTH_REPORT)

    failed = []
    for test_func in test_functions:
        try:
            test_func()
            print(f"  {test_func.__name__} passed")
        except AssertionError as e:
            failed.append((test_func.__name__, str(e)))
            print(f"  {test_func.__name__} failed: {e}")
        except Exception as e:
            failed.append((test_func.__name__, f"Test crashed: {e}"))
            print(f"  {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("TEST SUMMARY")
    print("=" * SEPARATOR_WIDTH_REPORT)
    print(f"Total tests: {len(test_functions)}")
    print(f"Passed: {len(test_functions) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")

    if not failed:
        print("\nALL TESTS PASSED! _load_from_csv_json() function verified completely.")
        return 0
    else:
        print(f"\nSome tests failed. Review the output above.")
        return 1
