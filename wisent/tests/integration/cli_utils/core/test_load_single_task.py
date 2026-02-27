#!/usr/bin/env python3
"""
Test suite for _load_single_task function in cli_prepare_dataset.py
Tests loading single benchmark tasks, similar to test_process_group_task.py
"""

import sys
from pathlib import Path
import torch
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent.cli_utils.cli_prepare_dataset import (
    _load_single_task,
    PrepState,
)
from wisent.core.model import Model
from wisent.core.constants import DEFAULT_SPLIT_RATIO
from wisent.tests.cli_utils._load_single_helpers import (
    test_load_single_task_with_shots,
    test_load_single_task_different_benchmarks,
    run_all_tests,
    cleanup_test_cache,
    TEST_CACHE_DIR,
)


def test_load_single_task_basic():
    """Test loading a single task with basic configuration"""

    print("="*80)
    print("Test 1: Load single task - basic configuration")
    print("="*80)

    cleanup_test_cache()

    # Check if CUDA is available and use appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device for testing")
    else:
        device = "cpu"
        print(f"CUDA not available, using CPU for testing")

    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B-Instruct", device=device)
        print("   Model created successfully")
    except Exception as e:
        print(f"   Failed to create model: {e}")
        print("   Test cannot continue without model")
        return False

    # Test loading a single task
    task_name = "winogrande"
    print(f"\n2. Loading task '{task_name}'...")

    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=0,
            split_ratio=DEFAULT_SPLIT_RATIO,
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

        print("\n3. Task loaded successfully")

        assert isinstance(result, PrepState), "Should return PrepState object"
        print("   Returned PrepState object")

        assert result.group_processed == False
        assert result.group_qa_format == False
        assert result.task_data is not None
        assert result.train_docs is not None

        print(f"   QA pairs extracted: {len(result.qa_pairs)}")
        print(f"   Test source: {len(result.test_source)}")
        print(f"   Train docs: {len(result.train_docs)}")

        print("\nTest passed")
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()


def test_load_single_task_with_limits():
    """Test loading a single task with training and testing limits"""

    print("\n" + "="*80)
    print("Test 2: Load single task with training and testing limits")
    print("="*80)

    cleanup_test_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B-Instruct", device=device)
        print("   Model created successfully")
    except Exception as e:
        print(f"   Failed to create model: {e}")
        return False

    task_name = "winogrande"
    training_limit = 8
    testing_limit = 2

    print(f"\n2. Loading task '{task_name}' with limits...")

    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=0,
            split_ratio=DEFAULT_SPLIT_RATIO,
            seed=42,
            limit=None,
            training_limit=training_limit,
            testing_limit=testing_limit,
            cache_benchmark=False,
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )

        print("\n3. Task loaded with limits")

        assert len(result.train_docs) <= training_limit
        assert len(result.test_source) <= testing_limit

        print(f"   Training docs: {len(result.train_docs)} (limit: {training_limit})")
        print(f"   Test source: {len(result.test_source)} (limit: {testing_limit})")

        print("\nTest passed")
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()


def test_load_single_task_with_cache():
    """Test loading a single task with cache_benchmark enabled"""

    print("\n" + "="*80)
    print("Test 3: Load single task with cache_benchmark enabled")
    print("="*80)

    cleanup_test_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B-Instruct", device=device)
        print("   Model created successfully")
    except Exception as e:
        print(f"   Failed to create model: {e}")
        return False

    task_name = "winogrande"

    print(f"\n2. Loading task '{task_name}' with cache_benchmark=True...")

    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=0,
            split_ratio=DEFAULT_SPLIT_RATIO,
            seed=42,
            limit=10,
            training_limit=None,
            testing_limit=None,
            cache_benchmark=True,
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )

        print("\n3. Task loaded (cache_benchmark=True)")

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


if __name__ == "__main__":
    test_functions = [
        test_load_single_task_basic,
        test_load_single_task_with_limits,
        test_load_single_task_with_cache,
        test_load_single_task_different_benchmarks,
        test_load_single_task_with_shots,
    ]
    success = run_all_tests(test_functions)
    exit(0 if success else 1)
