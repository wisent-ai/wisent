#!/usr/bin/env python3
"""
Test suite for _split_and_cap function in cli_prepare_dataset.py
Tests data splitting and capping functionality.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent.cli_utils.cli_prepare_dataset import _split_and_cap
from wisent.core.constants import DEFAULT_SPLIT_RATIO

# Import additional tests and runner from extracted module
from wisent.tests.cli_utils._split_cap_helpers import (
    test_split_and_cap_deterministic,
    test_split_and_cap_edge_cases,
    test_split_and_cap_different_ratios,
    test_split_and_cap_preserves_data,
    run_all_tests,
)


@dataclass
class Caps:
    train: int
    test: int


def test_split_and_cap_basic():
    """Test basic splitting with 80/20 ratio"""
    items = [{"id": i, "text": f"Item {i}"} for i in range(100)]

    train, test = _split_and_cap(
        items=items, split_ratio=DEFAULT_SPLIT_RATIO,
        caps=Caps(train=1000, test=1000),
        seed=42, verbose=True,
    )

    assert len(train) == 80, f"Expected 80 training items, got {len(train)}"
    assert len(test) == 20, f"Expected 20 test items, got {len(test)}"
    assert len(train) + len(test) == 100, "Total should equal input"


def test_split_and_cap_with_train_cap():
    """Test splitting with training cap applied"""
    items = [{"id": i, "text": f"Item {i}"} for i in range(100)]

    train, test = _split_and_cap(
        items=items, split_ratio=DEFAULT_SPLIT_RATIO,
        caps=Caps(train=50, test=1000),
        seed=42, verbose=True,
    )

    assert len(train) == 50, f"Expected 50 training items (capped), got {len(train)}"
    assert len(test) == 20, f"Expected 20 test items, got {len(test)}"


def test_split_and_cap_with_test_cap():
    """Test splitting with test cap applied"""
    items = [{"id": i, "text": f"Item {i}"} for i in range(100)]

    train, test = _split_and_cap(
        items=items, split_ratio=DEFAULT_SPLIT_RATIO,
        caps=Caps(train=1000, test=10),
        seed=42, verbose=True,
    )

    assert len(train) == 80, f"Expected 80 training items, got {len(train)}"
    assert len(test) == 10, f"Expected 10 test items (capped), got {len(test)}"


def test_split_and_cap_both_caps():
    """Test splitting with both caps applied"""
    items = [{"id": i, "text": f"Item {i}"} for i in range(100)]

    train, test = _split_and_cap(
        items=items, split_ratio=DEFAULT_SPLIT_RATIO,
        caps=Caps(train=30, test=5),
        seed=42, verbose=True,
    )

    assert len(train) == 30, f"Expected 30 training items (capped), got {len(train)}"
    assert len(test) == 5, f"Expected 5 test items (capped), got {len(test)}"


if __name__ == "__main__":
    success = run_all_tests([
        test_split_and_cap_basic,
        test_split_and_cap_with_train_cap,
        test_split_and_cap_with_test_cap,
        test_split_and_cap_both_caps,
        test_split_and_cap_deterministic,
        test_split_and_cap_edge_cases,
        test_split_and_cap_different_ratios,
        test_split_and_cap_preserves_data,
    ])
    exit(0 if success else 1)
