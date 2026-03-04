"""
Task-agnostic interface for benchmark integration.

This module provides a unified interface for integrating different benchmarks
without depending on lm-evaluation-harness.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from wisent.core.utils.services.benchmarks import BenchmarkExtractor
from wisent.core.utils.infra_tools.errors import TaskNotFoundError


class TaskInterface(ABC):
    """Abstract interface for benchmark tasks."""

    @abstractmethod
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load task data."""

    @abstractmethod
    def get_extractor(self) -> BenchmarkExtractor:
        """Get the benchmark extractor for this task."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the task name."""

    @abstractmethod
    def get_description(self) -> str:
        """Get the task description."""

    @abstractmethod
    def get_categories(self) -> List[str]:
        """Get the task categories (e.g., ['coding', 'reasoning'])."""


class TaskRegistry:
    """Registry for managing available tasks."""

    def __init__(self):
        self._tasks: Dict[str, Type[TaskInterface]] = {}

    def register_task(self, name: str, task_class: Type[TaskInterface]):
        """Register a new task."""
        self._tasks[name] = task_class

    def get_task(self, name: str, limit: Optional[int] = None, *, train_ratio: float) -> TaskInterface:
        """Get a task instance by name."""
        if name not in self._tasks:
            raise TaskNotFoundError(task_name=name, available_tasks=list(self._tasks.keys()))

        task_factory = self._tasks[name]

        # Handle different task factory types
        if callable(task_factory):
            # Try calling with limit parameter
            try:
                return task_factory(limit=limit, train_ratio=train_ratio)
            except TypeError:
                return task_factory(train_ratio=train_ratio)
        else:
            return task_factory(train_ratio=train_ratio)

    def list_tasks(self) -> List[str]:
        """List all available task names."""
        return list(self._tasks.keys())

    def get_task_info(self, name: str, *, train_ratio: float) -> Dict[str, Any]:
        """Get information about a specific task."""
        task = self.get_task(name, train_ratio=train_ratio)
        return {"name": task.get_name(), "description": task.get_description(), "categories": task.get_categories()}

    def list_task_info(self, *, train_ratio: float) -> List[Dict[str, Any]]:
        """List information about all available tasks."""
        return [self.get_task_info(name, train_ratio=train_ratio) for name in self.list_tasks()]


# Global task registry instance
_task_registry = TaskRegistry()


def register_task(name: str, task_class: Type[TaskInterface]):
    """Register a new task globally."""
    _task_registry.register_task(name, task_class)


def get_task(name: str, limit: Optional[int] = None, *, train_ratio: float) -> TaskInterface:
    """Get a task instance by name."""
    _ensure_tasks_registered()

    if "/" in name or "\\" in name or name.endswith(".json"):
        from .tasks.file_task import FileTask
        return FileTask(name, limit=limit)

    try:
        return _task_registry.get_task(name, limit=limit, train_ratio=train_ratio)
    except (ValueError, TaskNotFoundError):
        try:
            from .tasks.lm_eval_task import LMEvalTask, get_extractor
            get_extractor(name)
            return LMEvalTask(
                task_name=name,
                description=f"LM-eval task: {name}",
                categories=["lm-eval"],
                train_ratio=train_ratio,
            )
        except Exception:
            raise TaskNotFoundError(task_name=name, available_tasks=list(_task_registry._tasks.keys()))


def list_tasks() -> List[str]:
    """List all available task names."""
    _ensure_tasks_registered()
    return _task_registry.list_tasks()


def get_task_info(name: str, *, train_ratio: float) -> Dict[str, Any]:
    """Get information about a specific task."""
    return _task_registry.get_task_info(name, train_ratio=train_ratio)


def list_task_info(*, train_ratio: float) -> List[Dict[str, Any]]:
    """List information about all available tasks."""
    return _task_registry.list_task_info(train_ratio=train_ratio)


def _ensure_tasks_registered():
    """Ensure all tasks are registered in the global registry."""
    if len(_task_registry._tasks) == 0:  # Only register if not already done
        # Import tasks module to trigger registration
        # This is crucial for CLI usage where tasks module isn't imported elsewhere
        from . import tasks  # noqa: F401 # This imports __init__.py which calls register_all_tasks()
