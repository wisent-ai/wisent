
"""
Central registry for all available benchmarks.

This module provides a single source of truth for benchmark lists used across
the codebase. It loads from the parameter files:
- all_lm_eval_task_families.json (lm-eval tasks)
- not_lm_eval_tasks.json (HuggingFace-only tasks)
- broken_in_lm_eval.json (broken tasks to skip)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set

logger = logging.getLogger(__name__)

# Cache for loaded benchmarks
_benchmark_cache = {
    "all": None,
    "lm_eval": None,
    "huggingface_only": None,
    "broken": None,
    "working": None,
    "working_categories": None,
}


def _get_params_dir() -> Path:
    """Get the path to the lm_eval parameters directory."""
    # Navigate from this file: registry/ -> benchmarks/ -> services/ -> utils/ -> core/ -> wisent/
    this_dir = Path(__file__).resolve().parent
    wisent_root = this_dir.parent.parent.parent.parent.parent
    return wisent_root / "support" / "parameters" / "lm_eval"


def get_lm_eval_tasks() -> List[str]:
    """Get all lm-eval task families."""
    if _benchmark_cache["lm_eval"] is not None:
        return _benchmark_cache["lm_eval"]
    
    params_dir = _get_params_dir()
    lm_eval_tasks_path = params_dir / "all_lm_eval_task_families.json"
    
    lm_eval_tasks = []
    if lm_eval_tasks_path.exists():
        try:
            with open(lm_eval_tasks_path, 'r') as f:
                lm_eval_tasks = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load all_lm_eval_task_families.json: {e}")
    
    _benchmark_cache["lm_eval"] = lm_eval_tasks
    return lm_eval_tasks


def get_huggingface_only_tasks() -> List[str]:
    """Get tasks that are HuggingFace-only (not in lm-eval-harness)."""
    if _benchmark_cache["huggingface_only"] is not None:
        return _benchmark_cache["huggingface_only"]
    
    params_dir = _get_params_dir()
    not_lm_eval_tasks_path = params_dir / "not_lm_eval_tasks.json"
    
    not_lm_eval_tasks = []
    if not_lm_eval_tasks_path.exists():
        try:
            with open(not_lm_eval_tasks_path, 'r') as f:
                not_lm_eval_tasks = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load not_lm_eval_tasks.json: {e}")
    
    _benchmark_cache["huggingface_only"] = not_lm_eval_tasks
    return not_lm_eval_tasks


def get_huggingface_only_tasks_set() -> Set[str]:
    """Get HuggingFace-only tasks as a lowercase set for fast lookup."""
    return {t.lower() for t in get_huggingface_only_tasks()}


def get_broken_tasks() -> List[str]:
    """Get list of broken tasks to skip."""
    if _benchmark_cache["broken"] is not None:
        return _benchmark_cache["broken"]
    
    params_dir = _get_params_dir()
    broken_tasks_path = params_dir / "broken_in_lm_eval.json"
    
    broken_tasks = []
    if broken_tasks_path.exists():
        try:
            with open(broken_tasks_path, 'r') as f:
                broken_tasks = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load broken_in_lm_eval.json: {e}")
    
    _benchmark_cache["broken"] = broken_tasks
    return broken_tasks


def get_all_benchmarks() -> List[str]:
    """
    Get ALL available benchmarks, excluding broken ones.
    
    This combines:
    - all_lm_eval_task_families.json (lm-eval tasks)
    - not_lm_eval_tasks.json (HuggingFace-only tasks)
    - minus broken_in_lm_eval.json (broken tasks to skip)
    
    Returns:
        Sorted list of all available benchmark names
    """
    if _benchmark_cache["all"] is not None:
        return _benchmark_cache["all"]
    
    lm_eval_tasks = get_lm_eval_tasks()
    huggingface_only_tasks = get_huggingface_only_tasks()
    broken_tasks = set(get_broken_tasks())
    
    # Combine all tasks and filter out broken ones
    all_tasks = lm_eval_tasks + huggingface_only_tasks
    filtered_tasks = [task for task in all_tasks if task not in broken_tasks]
    
    result = sorted(filtered_tasks)
    _benchmark_cache["all"] = result
    return result


def load_all_benchmarks() -> Tuple[List[str], List[str]]:
    """
    Load all benchmarks and return both the filtered list and broken list.
    
    This is for backwards compatibility with train_unified_goodness.py.
    
    Returns:
        Tuple of (filtered_benchmarks, broken_benchmarks)
    """
    return get_all_benchmarks(), get_broken_tasks()


def is_huggingface_only_task(task_name: str) -> bool:
    """Check if a task is HuggingFace-only (not in lm-eval-harness)."""
    return task_name.lower() in get_huggingface_only_tasks_set()


def get_working_benchmarks() -> Set[str]:
    """Get the set of all working benchmarks from working_benchmarks_categorized.json."""
    if _benchmark_cache["working"] is not None:
        return _benchmark_cache["working"]
    params_dir = _get_params_dir()
    path = params_dir / "working_benchmarks_categorized.json"
    result = set()
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for tasks in data.values():
            result.update(tasks)
    _benchmark_cache["working"] = result
    return result


def get_working_benchmarks_with_categories() -> Dict[str, str]:
    """Get {benchmark: category} mapping from working_benchmarks_categorized.json."""
    if _benchmark_cache["working_categories"] is not None:
        return _benchmark_cache["working_categories"]
    params_dir = _get_params_dir()
    path = params_dir / "working_benchmarks_categorized.json"
    result = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for category, tasks in data.items():
            for task in tasks:
                result[task] = category
    _benchmark_cache["working_categories"] = result
    return result


def validate_benchmark(task_name: str) -> None:
    """Validate that a benchmark is in the working benchmarks list.

    Raises UnsupportedBenchmarkError if not found, with helpful context.
    """
    from wisent.core.utils.services.benchmarks.services.cache.download.managed_cached_benchmarks import (
        UnsupportedBenchmarkError,
    )
    working = get_working_benchmarks()
    if task_name in working:
        return
    lower_map = {t.lower(): t for t in working}
    if task_name.lower() in lower_map:
        return
    broken = set(get_broken_tasks())
    if task_name in broken:
        raise UnsupportedBenchmarkError(
            f"Benchmark '{task_name}' is known broken. "
            f"See broken_in_lm_eval.json."
        )
    raise UnsupportedBenchmarkError(
        f"Unknown benchmark '{task_name}'. "
        f"Valid benchmarks: {len(working)} in "
        f"working_benchmarks_categorized.json"
    )


def clear_cache():
    """Clear the benchmark cache (useful for testing)."""
    global _benchmark_cache
    _benchmark_cache = {
        "all": None,
        "lm_eval": None,
        "huggingface_only": None,
        "broken": None,
        "working": None,
        "working_categories": None,
    }
