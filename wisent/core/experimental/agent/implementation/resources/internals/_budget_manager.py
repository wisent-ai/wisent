"""BudgetManager for tracking resource budgets."""
import time
import logging
from typing import Dict, List, Optional, Any
from wisent.core.utils.infra_tools.errors import BudgetCalculationError, NoBenchmarkDataError, ResourceEstimationError
from wisent.core.experimental.agent.resources._budget_types import (
    ResourceType, ResourceBudget, TaskEstimate,
)
from wisent.core.utils.config_tools.constants import SECONDS_PER_MINUTE

logger = logging.getLogger(__name__)

class BudgetManager:
    """Manages budgets and resource allocation for agent operations."""

    def __init__(self, ema_alpha: float = None, default_quantity: int = None):
        self._ema_alpha = ema_alpha
        self._default_quantity = default_quantity
        self.budgets: Dict[ResourceType, ResourceBudget] = {}
        self.task_estimates: Dict[str, TaskEstimate] = {}
        self._default_estimates = self._get_default_task_estimates()
    
    def set_time_budget(self, minutes: float) -> None:
        """Set a time budget in minutes."""
        self.budgets[ResourceType.TIME] = ResourceBudget(
            resource_type=ResourceType.TIME,
            total_budget=minutes * SECONDS_PER_MINUTE,  # Convert to seconds
            unit="seconds"
        )
    
    def set_budget(self, resource_type: ResourceType, amount: float, unit: str = "") -> None:
        """Set a budget for any resource type."""
        self.budgets[resource_type] = ResourceBudget(
            resource_type=resource_type,
            total_budget=amount,
            unit=unit
        )
    
    def get_budget(self, resource_type: ResourceType) -> Optional[ResourceBudget]:
        """Get budget for a specific resource type."""
        return self.budgets.get(resource_type)
    
    def optimize_task_allocation(self, 
                               task_candidates: List[str],
                               primary_resource: ResourceType = ResourceType.TIME,
                               max_tasks: Optional[int] = None) -> List[str]:
        """
        Optimize task allocation within budget constraints.
        
        Args:
            task_candidates: List of candidate task names
            primary_resource: Primary resource to optimize for
            max_tasks: Maximum number of tasks to select
            
        Returns:
            List of selected tasks that fit within budget
        """
        budget = self.budgets.get(primary_resource)
        if not budget:
            return task_candidates[:max_tasks] if max_tasks else task_candidates
        
        # Calculate cost for each task
        task_costs = []
        for task in task_candidates:
            cost = self._estimate_task_cost(task, primary_resource)
            if cost > 0:
                task_costs.append((task, cost))
        
        # Sort by cost (ascending) to prioritize cheaper tasks
        task_costs.sort(key=lambda x: x[1])
        
        # Select tasks that fit within budget
        selected_tasks = []
        remaining_budget = budget.remaining_budget
        
        for task, cost in task_costs:
            if cost <= remaining_budget:
                selected_tasks.append(task)
                remaining_budget -= cost
                
                if max_tasks and len(selected_tasks) >= max_tasks:
                    break
        
        return selected_tasks
    
    def calculate_max_tasks_for_budget(self,
                                     task_type: str,
                                     time_budget_minutes: float = None) -> int:
        """
        Calculate maximum number of tasks that can fit within a time budget.

        Args:
            task_type: Type of task to estimate
            time_budget_minutes: Time budget in minutes

        Returns:
            Maximum number of tasks
        """
        if time_budget_minutes is None:
            raise ValueError("time_budget_minutes is required")
        time_budget_seconds = time_budget_minutes * SECONDS_PER_MINUTE

        # Get estimate for this task type
        task_estimate = self._estimate_task_cost(task_type, ResourceType.TIME)
        
        if task_estimate <= 0:
            return 1  # Fallback to at least 1 task
        
        max_tasks = max(1, int(time_budget_seconds / task_estimate))
        return max_tasks
    
    def estimate_completion_time(self, tasks: List[str]) -> float:
        """
        Estimate total completion time for a list of tasks.
        
        Args:
            tasks: List of task names
            
        Returns:
            Estimated time in seconds
        """
        total_time = 0.0
        for task in tasks:
            total_time += self._estimate_task_cost(task, ResourceType.TIME)
        return total_time
    
    def track_task_execution(self, task_name: str, start_time: float, end_time: float) -> None:
        """
        Track actual execution time for a task to improve future estimates.
        
        Args:
            task_name: Name of the task
            start_time: Start timestamp
            end_time: End timestamp
        """
        actual_time = end_time - start_time
        
        # Update our estimates based on actual performance
        if task_name in self.task_estimates:
            # Use exponential moving average to update estimates
            if self._ema_alpha is None:
                raise ValueError("ema_alpha is required for tracking task execution")
            current_estimate = self.task_estimates[task_name].time_seconds
            alpha = self._ema_alpha  # Learning rate
            new_estimate = alpha * actual_time + (1 - alpha) * current_estimate
            self.task_estimates[task_name].time_seconds = new_estimate
        else:
            # First time seeing this task
            self.task_estimates[task_name] = TaskEstimate(
                task_name=task_name,
                time_seconds=actual_time
            )
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get a summary of all budgets and their usage."""
        summary = {}
        for resource_type, budget in self.budgets.items():
            summary[resource_type.value] = {
                "total": budget.total_budget,
                "used": budget.used_budget,
                "remaining": budget.remaining_budget,
                "percentage_used": budget.usage_percentage,
                "unit": budget.unit
            }
        return summary
    
    def _estimate_task_cost(self, task_name: str, resource_type: ResourceType) -> float:
        """Estimate the cost of a task for a specific resource type."""
        # Check if we have a specific estimate for this task
        if task_name in self.task_estimates:
            estimate = self.task_estimates[task_name]
            if resource_type == ResourceType.TIME:
                return estimate.time_seconds
            elif resource_type == ResourceType.MEMORY:
                return estimate.memory_mb
            elif resource_type == ResourceType.COMPUTE:
                return estimate.compute_units
            elif resource_type == ResourceType.TOKENS:
                return float(estimate.tokens)
        
        # Fall back to default estimates
        return self._get_default_cost_estimate(task_name, resource_type)
    
    def _get_default_cost_estimate(self, task_name: str, resource_type: ResourceType) -> float:
        """Get default cost estimate for a task using device benchmarking."""
        if resource_type == ResourceType.TIME and self._default_quantity is None:
            raise ValueError("default_quantity is required for cost estimation")
        if resource_type == ResourceType.TIME:
            # Use device-specific benchmarks for time estimates
            try:
                from .device_benchmarks import estimate_task_time
                
                # Map task names to benchmark types
                task_mapping = {
                    "benchmark": "benchmark_eval",
                    "eval": "benchmark_eval", 
                    "classifier": "classifier_training",
                    "training": "classifier_training",
                    "generation": "data_generation",
                    "synthetic": "data_generation",
                    "steering": "steering",
                    "model_loading": "model_loading"
                }
                
                # Find the best matching task type
                benchmark_type = None
                for pattern, task_type in task_mapping.items():
                    if pattern in task_name.lower():
                        benchmark_type = task_type
                        break
                
                if benchmark_type:
                    # Get quantity based on task type
                    if benchmark_type in ["benchmark_eval", "classifier_training"]:
                        quantity = self._default_quantity  # Base unit for these tasks
                    else:
                        quantity = 1
                    
                    return estimate_task_time(benchmark_type, quantity)
                else:
                    # Use benchmark_eval as default
                    return estimate_task_time("benchmark_eval", self._default_quantity)
                    
            except Exception as e:
                raise ResourceEstimationError(resource_type="time", task_name=task_name, cause=e)
        
        elif resource_type == ResourceType.MEMORY:
            raise ResourceEstimationError(resource_type="memory", task_name=task_name)
        
        elif resource_type == ResourceType.COMPUTE:
            raise ResourceEstimationError(resource_type="compute", task_name=task_name)
        
        elif resource_type == ResourceType.TOKENS:
            raise ResourceEstimationError(resource_type="tokens", task_name=task_name)
        
        raise UnknownTypeError(entity_type="resource_type", value=str(resource_type))
    
    def _get_default_task_estimates(self) -> Dict[str, TaskEstimate]:
        """Get default task estimates for common operations."""
        # No default estimates - all estimates must come from device benchmarks
        return {}


# Global budget manager instance
_budget_manager = BudgetManager()


