"""Budget convenience functions."""
import time
import logging
from typing import Dict, List, Optional, Any
from wisent.core.utils.infra_tools.errors import BudgetCalculationError, NoBenchmarkDataError, ResourceEstimationError
from wisent.core.experimental.agent.resources._budget_types import ResourceType, ResourceBudget, TaskEstimate
from wisent.core.experimental.agent.resources._budget_manager import BudgetManager
from wisent.core.utils.config_tools.constants import (
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    SECONDS_PER_MINUTE,
)
logger = logging.getLogger(__name__)

def get_budget_manager() -> BudgetManager:
    """Get the global budget manager instance."""
    return _budget_manager


def set_time_budget(minutes: float) -> None:
    """Convenience function to set time budget."""
    _budget_manager.set_time_budget(minutes)


def calculate_max_tasks_for_time_budget(task_type: str,
                                       time_budget_minutes: float = None) -> int:
    """
    Calculate maximum number of tasks that can fit within a time budget.

    Args:
        task_type: Type of task to estimate (benchmark_evaluation, classifier_training, etc.)
        time_budget_minutes: Time budget in minutes

    Returns:
        Maximum number of tasks
    """
    if time_budget_minutes is None:
        raise ValueError("time_budget_minutes is required")
    # Use device benchmarking for more accurate estimates
    try:
        from .device_benchmarks import estimate_task_time
        
        # Map task types to benchmark types
        benchmark_mapping = {
            "benchmark_evaluation": "benchmark_eval",
            "classifier_training": "classifier_training",
            "data_generation": "data_generation",
            "steering": "steering",
            "model_loading": "model_loading"
        }
        
        benchmark_type = benchmark_mapping.get(task_type, "benchmark_eval")
        
        # Get time per task
        if benchmark_type in ["benchmark_eval", "classifier_training"]:
            time_per_task = estimate_task_time(benchmark_type, 100) / 100  # Per unit
        else:
            time_per_task = estimate_task_time(benchmark_type, 1)
        
        time_budget_seconds = time_budget_minutes * SECONDS_PER_MINUTE
        max_tasks = max(1, int(time_budget_seconds / time_per_task))
        
        return max_tasks
        
    except Exception as e:
        raise BudgetCalculationError(task_type=task_type, cause=e)


def optimize_tasks_for_budget(task_candidates: List[str],
                            time_budget_minutes: float = None,
                            max_tasks: Optional[int] = None) -> List[str]:
    """
    Optimize task selection within a time budget.

    Args:
        task_candidates: List of candidate task names
        time_budget_minutes: Time budget in minutes
        max_tasks: Maximum number of tasks to select

    Returns:
        List of selected tasks that fit within budget
    """
    if time_budget_minutes is None:
        raise ValueError("time_budget_minutes is required")
    _budget_manager.set_time_budget(time_budget_minutes)
    return _budget_manager.optimize_task_allocation(
        task_candidates, 
        ResourceType.TIME, 
        max_tasks
    )


def optimize_benchmarks_for_budget(task_candidates: List[str],
                                 time_budget_minutes: float = None,
                                 max_tasks: Optional[int] = None,
                                 prefer_fast: bool = False) -> List[str]:
    """
    Optimize benchmark selection within a time budget using priority and loading time data.

    Args:
        task_candidates: List of candidate benchmark names
        time_budget_minutes: Time budget in minutes
        max_tasks: Maximum number of tasks to select
        prefer_fast: Whether to prefer fast benchmarks

    Returns:
        List of selected benchmarks that fit within budget
    """
    if time_budget_minutes is None:
        raise ValueError("time_budget_minutes is required")
    try:
        # Import benchmark data
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm-harness-integration'))
        from only_benchmarks import BENCHMARKS
        
        # Get benchmark information with loading times
        benchmark_info = []
        for task in task_candidates:
            if task in BENCHMARKS:
                config = BENCHMARKS[task]
                loading_time = config['loading_time']  # seconds
                priority = config.get('priority', 'unknown')
                
                # Calculate priority score for selection
                priority_score = 0
                if priority == 'high':
                    priority_score = PRIORITY_HIGH
                elif priority == 'medium':
                    priority_score = PRIORITY_MEDIUM
                elif priority == 'low':
                    priority_score = PRIORITY_LOW
                
                # Calculate efficiency score (priority per second)
                efficiency_score = priority_score / max(loading_time, 1.0)
                
                benchmark_info.append({
                    'task': task,
                    'loading_time': loading_time,
                    'priority': priority,
                    'priority_score': priority_score,
                    'efficiency_score': efficiency_score
                })
            else:
                raise NoBenchmarkDataError()
        
        # Sort by efficiency (prefer fast) or priority (prefer high priority)
        if prefer_fast:
            benchmark_info.sort(key=lambda x: x['efficiency_score'], reverse=True)
        else:
            benchmark_info.sort(key=lambda x: (x['priority_score'], -x['loading_time']), reverse=True)
        
        # Select benchmarks that fit within budget
        selected_benchmarks = []
        total_time = 0.0
        time_budget_seconds = time_budget_minutes * SECONDS_PER_MINUTE

        for info in benchmark_info:
            if total_time + info['loading_time'] <= time_budget_seconds:
                selected_benchmarks.append(info['task'])
                total_time += info['loading_time']
                
                if max_tasks and len(selected_benchmarks) >= max_tasks:
                    break
        
        return selected_benchmarks
        
    except Exception as e:
        print(f"   ⚠️ Priority-aware budget optimization failed: {e}")
        print(f"   🔄 Falling back to basic budget optimization...")
        return optimize_tasks_for_budget(task_candidates, time_budget_minutes, max_tasks)


