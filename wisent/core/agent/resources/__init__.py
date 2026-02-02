"""Resource management and device benchmarking for wisent agent."""

from .budget import (
    ResourceType,
    ResourceBudget,
    TaskEstimate,
    BudgetManager,
    get_budget_manager,
    set_time_budget,
    calculate_max_tasks_for_time_budget,
    optimize_tasks_for_budget,
    optimize_benchmarks_for_budget,
    estimate_completion_time_minutes,
    track_task_performance,
    run_device_benchmark,
    get_device_info,
    estimate_task_time_direct,
)
from .device_benchmarks import (
    DeviceBenchmark,
    DeviceBenchmarker,
    get_device_benchmarker,
    ensure_benchmark_exists,
    estimate_task_time,
    get_current_device_info,
)

__all__ = [
    # Budget management
    'ResourceType',
    'ResourceBudget',
    'TaskEstimate',
    'BudgetManager',
    'get_budget_manager',
    'set_time_budget',
    'calculate_max_tasks_for_time_budget',
    'optimize_tasks_for_budget',
    'optimize_benchmarks_for_budget',
    'estimate_completion_time_minutes',
    'track_task_performance',
    'run_device_benchmark',
    'get_device_info',
    'estimate_task_time_direct',
    # Device benchmarks
    'DeviceBenchmark',
    'DeviceBenchmarker',
    'get_device_benchmarker',
    'ensure_benchmark_exists',
    'estimate_task_time',
    'get_current_device_info',
]
