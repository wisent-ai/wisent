"""
Budget and resource management for wisent agent operations.

This module provides utilities for managing time budgets, resource allocation,
and optimizing task execution within specified constraints.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import math

from wisent.core.errors import (
    ResourceEstimationError,
    BudgetCalculationError,
    UnknownTypeError,
)


class ResourceType(Enum):
    """Types of resources that can be budgeted."""
    TIME = "time"
    MEMORY = "memory" 
    COMPUTE = "compute"
    TOKENS = "tokens"


@dataclass
class ResourceBudget:
    """Represents a budget for a specific resource type."""
    resource_type: ResourceType
    total_budget: float
    used_budget: float = 0.0
    unit: str = ""
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return max(0.0, self.total_budget - self.used_budget)
    
    @property
    def usage_percentage(self) -> float:
        """Calculate percentage of budget used."""
        if self.total_budget <= 0:
            return 0.0
        return (self.used_budget / self.total_budget) * 100.0
    
    def can_afford(self, cost: float) -> bool:
        """Check if we can afford a given cost."""
        return self.remaining_budget >= cost
    
    def spend(self, amount: float) -> bool:
        """Spend from the budget. Returns True if successful."""
        if self.can_afford(amount):
            self.used_budget += amount
            return True
        return False


@dataclass
class TaskEstimate:
    """Estimates for a specific task."""
    task_name: str
    time_seconds: float
    memory_mb: float = 0.0
    compute_units: float = 0.0
    tokens: int = 0
    
    def scale(self, factor: float) -> 'TaskEstimate':
        """Scale all estimates by a factor."""
        return TaskEstimate(
            task_name=self.task_name,
            time_seconds=self.time_seconds * factor,
            memory_mb=self.memory_mb * factor,
            compute_units=self.compute_units * factor,
            tokens=int(self.tokens * factor)
        )


