"""Extracted split-and-cap test functions and runner."""

from dataclasses import dataclass


@dataclass
class Caps:
    train: int
    test: int


def _get_split_fn():
    """Late import to avoid circular dependency issues."""
    from wisent.cli_utils.cli_prepare_dataset import _split_and_cap
    return _split_and_cap


def test_split_and_cap_deterministic():
    """Test 5: Deterministic splitting with seed"""
    _split_and_cap = _get_split_fn()

    items = [{"id": i, "text": f"Item {i}"} for i in range(50)]

    # First split
    train1, test1 = _split_and_cap(
        items=items, split_ratio=0.7, caps=Caps(train=1000, test=1000),
        seed=123, verbose=False,
    )
    # Second split with same seed
    train2, test2 = _split_and_cap(
        items=items, split_ratio=0.7, caps=Caps(train=1000, test=1000),
        seed=123, verbose=False,
    )
    # Third split with different seed
    train3, test3 = _split_and_cap(
        items=items, split_ratio=0.7, caps=Caps(train=1000, test=1000),
        seed=456, verbose=False,
    )

    assert train1 == train2, "Same seed should produce identical training sets"
    assert test1 == test2, "Same seed should produce identical test sets"
    assert train1 != train3, "Different seeds should produce different training sets"


def test_split_and_cap_edge_cases():
    """Test 6: Edge cases: empty list, single item, extreme ratios"""
    _split_and_cap = _get_split_fn()

    # Test 1: Empty list
    train, test = _split_and_cap(
        items=[], split_ratio=0.8, caps=Caps(train=100, test=100),
        seed=42, verbose=False,
    )
    assert len(train) == 0 and len(test) == 0, "Empty input should produce empty outputs"

    # Test 2: Single item with 0.5 split
    train, test = _split_and_cap(
        items=[{"id": 1}], split_ratio=0.5, caps=Caps(train=100, test=100),
        seed=42, verbose=False,
    )
    assert len(train) + len(test) == 1, "Single item should be in either train or test"

    # Test 3: Extreme split ratio (0.0)
    items = [{"id": i} for i in range(10)]
    train, test = _split_and_cap(
        items=items, split_ratio=0.0, caps=Caps(train=100, test=100),
        seed=42, verbose=False,
    )
    assert len(train) == 0, "Ratio 0.0 should put all in test"
    assert len(test) == 10, "Ratio 0.0 should put all in test"

    # Test 4: Extreme split ratio (1.0)
    train, test = _split_and_cap(
        items=items, split_ratio=1.0, caps=Caps(train=100, test=100),
        seed=42, verbose=False,
    )
    assert len(train) == 10, "Ratio 1.0 should put all in train"
    assert len(test) == 0, "Ratio 1.0 should put all in train"


def test_split_and_cap_different_ratios():
    """Test 7: Various split ratios"""
    _split_and_cap = _get_split_fn()

    items = [{"id": i} for i in range(100)]
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9]

    for ratio in ratios:
        train, test = _split_and_cap(
            items=items, split_ratio=ratio, caps=Caps(train=1000, test=1000),
            seed=42, verbose=False,
        )
        expected_train = int(100 * ratio)
        expected_test = 100 - expected_train
        assert len(train) == expected_train, f"Incorrect train size for ratio {ratio}"
        assert len(test) == expected_test, f"Incorrect test size for ratio {ratio}"


def test_split_and_cap_preserves_data():
    """Test 8: Data preservation (no loss or duplication)"""
    _split_and_cap = _get_split_fn()

    items = [{"id": i, "value": f"val_{i}"} for i in range(73)]  # Odd number

    train, test = _split_and_cap(
        items=items, split_ratio=0.7, caps=Caps(train=1000, test=1000),
        seed=999, verbose=False,
    )

    # Check no data loss
    total_after = len(train) + len(test)
    assert total_after == len(items), \
        f"Data lost! Had {len(items)}, now have {total_after}"

    # Check no duplication
    all_ids = [item['id'] for item in train + test]
    unique_ids = set(all_ids)
    assert len(all_ids) == len(unique_ids), "Duplicate items found!"

    # Check all original IDs are present
    original_ids = set(item['id'] for item in items)
    split_ids = set(item['id'] for item in train + test)
    assert original_ids == split_ids, "Some IDs missing or changed!"


def run_all_tests(test_functions):
    """Run all tests."""
    print("\nRunning _split_and_cap tests...")
    print("=" * 80)

    failed = []
    for test_func in test_functions:
        try:
            test_func()
            print(f"  {test_func.__name__} passed")
        except Exception as e:
            print(f"  {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)

    print("\n" + "=" * 80)
    if failed:
        print(f"{len(failed)} test(s) failed: {', '.join(failed)}")
        return False
    else:
        print("All tests passed!")
        return True
