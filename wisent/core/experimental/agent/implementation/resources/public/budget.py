"""Budget management for wisent agent resource allocation.

Re-exports from split modules for backward compatibility.
"""
from wisent.core.experimental.agent.resources._budget_types import (
    ResourceType, ResourceBudget, TaskEstimate,
)
from wisent.core.experimental.agent.resources._budget_manager import BudgetManager
from wisent.core.experimental.agent.resources._budget_functions import (
    get_budget_manager,
    set_time_budget,
    calculate_max_tasks_for_time_budget,
    optimize_tasks_for_budget,
    optimize_benchmarks_for_budget,
)
from wisent.core.experimental.agent.resources._budget_device import (
    estimate_completion_time_minutes,
    track_task_performance,
    run_device_benchmark,
    get_device_info,
    estimate_task_time_direct,
    main,
)
