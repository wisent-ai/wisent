#!/usr/bin/env python3
"""
Comprehensive test for _load_from_csv_json() function in cli_prepare_dataset.py
Tests CSV and JSON loading with various conditions and edge cases.
"""

import json
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent.cli_utils.cli_prepare_dataset import (
    _load_from_csv_json,
    PrepState,
    Caps
)
from wisent.core.primitives.contrastive_pairs.contrastive_pair_set import ContrastivePairSet

# Import helpers and additional tests from extracted module
from wisent.tests.cli_utils._csv_json_helpers import (
    create_test_csv,
    create_test_json,
    test_deterministic_split,
    test_different_column_names,
    test_split_ratios,
    run_all_tests,
)


def test_csv_loading_disabled():
    """Test 1: CSV loading when from_csv=False"""
    result = _load_from_csv_json(
        from_csv=False,
        from_json=False,
        task_name="dummy.csv",
        question_col="question",
        correct_col="correct",
        incorrect_col="incorrect",
        limit=None,
        caps=Caps(train=100, test=50),
        split_ratio=0.8,
        seed=42,
        verbose=False
    )

    assert result is None, f"Expected None when from_csv=False, got {type(result)}"


def test_csv_loading_basic():
    """Test 2: Basic CSV loading with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"

        # Create test data
        test_data = [
            {"question": "What is 2+2?", "correct_answer": "4", "incorrect_answer": "5"},
            {"question": "What is 3+3?", "correct_answer": "6", "incorrect_answer": "7"},
            {"question": "What is 4+4?", "correct_answer": "8", "incorrect_answer": "9"},
            {"question": "What is 5+5?", "correct_answer": "10", "incorrect_answer": "11"},
            {"question": "What is 6+6?", "correct_answer": "12", "incorrect_answer": "13"},
        ]
        create_test_csv(csv_path, test_data)

        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=True
        )

        # Verify result
        assert result is not None, "CSV loading returned None"
        assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"

        # Check data split (80/20)
        total_items = len(result.qa_pairs) + len(result.test_source)
        assert total_items == 5, f"Expected 5 items, got {total_items}"

        # Check split ratio (approximately 80/20)
        expected_train = 4  # 80% of 5
        expected_test = 1   # 20% of 5
        assert len(result.qa_pairs) == expected_train, \
            f"Expected {expected_train} train, got {len(result.qa_pairs)}"
        assert len(result.test_source) == expected_test, \
            f"Expected {expected_test} test, got {len(result.test_source)}"

        # Check flags
        assert result.group_processed == True, \
            f"group_processed should be True, got {result.group_processed}"
        assert result.group_qa_format == True, \
            f"group_qa_format should be True, got {result.group_qa_format}"


def test_json_loading_basic(split_ratio: float):
    """Test 3: Basic JSON loading with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test.json"

        # Create test data in the expected JSON format
        test_data = [
            {"question": f"Question {i}", "correct_answer": f"Answer {i}",
             "incorrect_answer": f"Wrong {i}"}
            for i in range(10)
        ]
        create_test_json(json_path, test_data)

        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=False,
            from_json=True,
            task_name=str(json_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=split_ratio,
            seed=42,
            verbose=False
        )

        # Verify result
        assert result is not None, "JSON loading returned None"
        assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"

        # Check data split
        total_items = len(result.qa_pairs) + len(result.test_source)
        assert total_items == 10, f"Expected 10 items, got {total_items}"

        # Check split
        expected_train = int(total_items * split_ratio)
        expected_test = total_items - expected_train
        assert len(result.qa_pairs) == expected_train, \
            f"Expected {expected_train} train, got {len(result.qa_pairs)}"
        assert len(result.test_source) == expected_test, \
            f"Expected {expected_test} test, got {len(result.test_source)}"

        # Check flags
        assert result.group_processed == True
        assert result.group_qa_format == True


def test_with_limits():
    """Test 4: CSV loading with limit parameter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"

        # Create large dataset
        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(100)
        ]
        create_test_csv(csv_path, test_data)

        # Only load first 20 items
        limit = None

        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=limit,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )

        # Check that limit was applied
        total = len(result.qa_pairs) + len(result.test_source)
        assert total == limit, f"Expected {limit} items, got {total}"


def test_with_caps():
    """Test 5: Loading with training and testing caps"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"

        # Create dataset
        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(50)
        ]
        create_test_csv(csv_path, test_data)

        # Set small caps
        train_cap = 5
        test_cap = 2

        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=train_cap, test=test_cap),
            split_ratio=0.8,
            seed=42,
            verbose=True
        )

        # Check caps were applied
        assert len(result.qa_pairs) <= train_cap, \
            f"Train cap exceeded: {len(result.qa_pairs)} > {train_cap}"
        assert len(result.test_source) <= test_cap, \
            f"Test cap exceeded: {len(result.test_source)} > {test_cap}"


def test_empty_file():
    """Test 6: Loading empty CSV file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "empty.csv"
        # Create empty CSV with headers only
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=["question", "correct_answer", "incorrect_answer"])
            writer.writeheader()

        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )

        assert result is not None, "Result is None for empty file"
        assert len(result.qa_pairs) == 0, \
            f"Expected 0 train items for empty file, got {len(result.qa_pairs)}"
        assert len(result.test_source) == 0, \
            f"Expected 0 test items for empty file, got {len(result.test_source)}"


if __name__ == "__main__":
    import argparse
    import functools
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-ratio", type=float, required=True, help="Train/test split ratio")
    cli_args = parser.parse_args()
    exit(run_all_tests([
        test_csv_loading_disabled,
        test_csv_loading_basic,
        functools.partial(test_json_loading_basic, split_ratio=cli_args.split_ratio),
        test_with_limits,
        test_with_caps,
        test_empty_file,
        test_deterministic_split,
        test_different_column_names,
        test_split_ratios,
    ]))
