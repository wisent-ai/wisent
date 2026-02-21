"""
Additional test functions for _load_single_task.
Extracted from test_load_single_task.py to keep file under 300 lines.
"""

import sys
from pathlib import Path
import torch
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent.cli_utils.cli_prepare_dataset import (
    _load_single_task,
    PrepState,
)
from wisent.core.model import Model


# Test cache directory
TEST_CACHE_DIR = "./benchmark_cache_test_load_single"


def cleanup_test_cache() -> None:
    """Remove test cache directory if it exists."""
    if Path(TEST_CACHE_DIR).exists():
        shutil.rmtree(TEST_CACHE_DIR)


def test_load_single_task_with_shots():
    """Test loading a single task with few-shot examples"""

    print("\n" + "="*80)
    print("Test 5: Load single task with few-shot examples")
    print("="*80)

    cleanup_test_cache()

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B-Instruct", device=device)
        print("   Model created successfully")
    except Exception as e:
        print(f"   Failed to create model: {e}")
        return False

    task_name = "winogrande"
    shots = 3

    print(f"\n2. Loading task '{task_name}' with {shots} shots...")

    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=shots,  # Few-shot examples
            split_ratio=0.8,
            seed=42,
            limit=10,
            training_limit=None,
            testing_limit=None,
            cache_benchmark=False,
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )

        print(f"\n3. Task loaded with {shots} shots")

        assert isinstance(result, PrepState), "Should return PrepState object"
        print("   Returned PrepState object")
        print(f"   QA pairs: {len(result.qa_pairs)}")
        print(f"   Test source: {len(result.test_source)}")

        print("\nTest passed")
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()


def test_load_single_task_different_benchmarks():
    """Test loading different benchmark tasks"""

    print("\n" + "="*80)
    print("Test 4: Load different benchmark tasks")
    print("="*80)

    cleanup_test_cache()

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B-Instruct", device=device)
        print("   Model created successfully")
    except Exception as e:
        print(f"   Failed to create model: {e}")
        return False

    # Test different benchmarks
    benchmarks = ["winogrande", "piqa", "copa"]

    print(f"\n2. Testing different benchmarks: {benchmarks}")

    for task_name in benchmarks:
        print(f"\n   Loading '{task_name}'...")
        try:
            result = _load_single_task(
                model=model,
                task_name=task_name,
                shots=0,
                split_ratio=0.8,
                seed=42,
                limit=5,  # Very small limit for speed
                training_limit=None,
                testing_limit=None,
                cache_benchmark=False,
                cache_dir=TEST_CACHE_DIR,
                force_download=False,
                livecodebench_version="v1",
                verbose=False  # Less verbose for multiple tasks
            )

            assert isinstance(result, PrepState), f"Failed to load {task_name}"
            print(f"   {task_name}: {len(result.qa_pairs)} QA pairs, {len(result.test_source)} test source")

        except Exception as e:
            print(f"   Failed to load {task_name}: {e}")
            return False

    print("\nAll benchmarks loaded successfully")
    print("Test passed")
    cleanup_test_cache()
    return True


def run_all_tests(test_functions):
    """Run all tests"""
    print("\nRunning _load_single_task tests...")
    print("="*80)

    failed = []

    for test_func in test_functions:
        try:
            success = test_func()
            if not success:
                failed.append(test_func.__name__)
        except Exception as e:
            print(f"\nTest {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)

    print("\n" + "="*80)
    if failed:
        print(f"{len(failed)} test(s) failed: {', '.join(failed)}")
        return False
    else:
        print("All tests passed!")
        return True
