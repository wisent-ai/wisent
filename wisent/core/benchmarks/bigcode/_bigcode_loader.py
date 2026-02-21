"""
BigCode Evaluation Harness integration for Wisent.

This module provides integration with bigcode-evaluation-harness for code generation benchmarks.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.errors import TaskNotFoundError, InsufficientDataError

logger = logging.getLogger(__name__)


class BigCodeTaskLoader:
    """Loads and manages BigCode evaluation tasks."""

    # Mapping of our task names to BigCode task names
    TASK_MAPPING = {
        # === DIRECT MATCHES ===
        "humaneval": "humaneval",
        "mbpp": "mbpp",
        "conala": "conala",
        "concode": "concode",
        "mercury": "mercury",
        # === CORRECTED MAPPINGS ===
        "humaneval_plus": "humanevalplus",
        "instructhumaneval": "instruct-humaneval",
        "mbpp_plus": "mbppplus",
        "apps": "apps-introductory",
        "ds1000": "ds1000-all-completion",
        # === MULTI-LANGUAGE TASKS ===
        "multiple_py": "multiple-py",
        "multiple_js": "multiple-js",
        "multiple_java": "multiple-java",
        "multiple_cpp": "multiple-cljcpp",
        "multiple_rs": "multiple-rs",
        "multiple_go": "multiple-go",
        # === CODE-TO-TEXT TASKS ===
        "codexglue_code_to_text_python": "codexglue_code_to_text-python",
        "codexglue_code_to_text_go": "codexglue_code_to_text-go",
        "codexglue_code_to_text_java": "codexglue_code_to_text-java",
        "codexglue_code_to_text_javascript": "codexglue_code_to_text-javascript",
        "codexglue_code_to_text_php": "codexglue_code_to_text-php",
        "codexglue_code_to_text_ruby": "codexglue_code_to_text-ruby",
        # === FIXED PROBLEMATIC MAPPINGS ===
        "recode": "perturbed-humaneval-natgen-num_seeds_1",
        "humanevalpack": None,  # ❌ REMOVED - no simple mapping exists, only complex variants
    }

    def __init__(self):
        """Initialize BigCode task loader."""
        self._bigcode_available = self._check_bigcode_available()
        self._task_cache = {}

    def _check_bigcode_available(self) -> bool:
        """Check if bigcode-evaluation-harness is available."""
        try:
            import bigcode_eval

            return True
        except ImportError:
            logger.warning("bigcode-evaluation-harness not installed")
            return False

    def is_bigcode_task(self, task_name: str) -> bool:
        """Check if a task is a BigCode task."""
        return task_name in self.TASK_MAPPING

    def load_task(self, task_name: str, limit: Optional[int] = None) -> "BigCodeTask":
        """
        Load a BigCode task.

        Args:
            task_name: Name of the task (our naming convention)
            limit: Optional limit on number of samples

        Returns:
            BigCodeTask object
        """
        if not self._bigcode_available:
            raise ImportError("bigcode-evaluation-harness not installed. Run: pip install bigcode-evaluation-harness")

        if task_name not in self.TASK_MAPPING:
            raise TaskNotFoundError(task_name=task_name, available_tasks=list(self.TASK_MAPPING.keys()))

        bigcode_task_name = self.TASK_MAPPING[task_name]

        # Handle removed tasks with None mapping
        if bigcode_task_name is None:
            raise TaskNotFoundError(task_name=task_name)

        # Check cache
        cache_key = f"{task_name}:{limit}"
        if cache_key in self._task_cache:
            return self._task_cache[cache_key]

        # Create task object
        task = BigCodeTask(task_name, bigcode_task_name, limit)
        self._task_cache[cache_key] = task

        return task


class BigCodeTask:
    """Represents a BigCode evaluation task."""

    def __init__(self, task_name: str, bigcode_task_name: str, limit: Optional[int] = None):
        """
        Initialize BigCode task.

        Args:
            task_name: Our task name
            bigcode_task_name: BigCode's task name
            limit: Optional limit on samples
        """
        self.task_name = task_name
        self.bigcode_task_name = bigcode_task_name
        self.limit = limit
        self._limit = limit  # Store as private attribute too
        self._data = None
        self._task_obj = None
        self._load_data()

    def _load_data(self):
        """Load task data from BigCode."""
        try:
            # Import BigCode modules
            from bigcode_eval.tasks import get_task

            # Get the task
            task = get_task(self.bigcode_task_name)
            self._task_obj = task

            # Get dataset - BigCode uses get_dataset() method
            dataset = task.get_dataset()

            # Convert to list if needed
            if hasattr(dataset, "__iter__"):
                dataset = list(dataset)

            # Apply limit if specified
            if self.limit:
                dataset = dataset[: self.limit]

            self._data = dataset

        except Exception as e:
            logger.error(f"Failed to load BigCode task {self.bigcode_task_name}: {e}")
            # Fallback to loading from files if available
            self._load_from_files()

    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # BigCode tasks don't have separate validation sets

    def has_test_docs(self) -> bool:
        """Check if task has test documents."""
        return True  # All samples are considered test docs

    def test_docs(self) -> List[Dict[str, Any]]:
        """Get test documents."""
        return self.get_samples()

    def validation_docs(self) -> List[Dict[str, Any]]:
        """Get validation documents."""
        return []  # No separate validation set

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text prompt."""
        # Handle different BigCode formats
        if "prompt" in doc:
            return doc["prompt"]
        if "text" in doc:
            return doc["text"]
        if "question" in doc:
            return doc["question"]
        if "problem" in doc:
            return doc["problem"]
        # Fallback - try to use task object if available
        if self._task_obj and hasattr(self._task_obj, "get_prompt"):
            return self._task_obj.get_prompt(doc)
        return str(doc)

    def _load_from_files(self):
        """Load task data from local files as fallback."""
        # Try to load from standard locations
        data_paths = [
            f"~/.cache/bigcode_eval/{self.bigcode_task_name}",
            f"data/{self.bigcode_task_name}",
            f"bigcode_eval/tasks/{self.bigcode_task_name}",
        ]

        for path in data_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                self._load_from_path(expanded_path)
                return

        # If no data found, raise error
        raise InsufficientDataError(reason=f"No data found for task {self.task_name}. Please provide valid benchmark data.")

    def _load_from_path(self, path: str):
        """Load data from a specific path."""
        data = []

        # Look for JSON/JSONL files
        for file in Path(path).glob("*.json*"):
            with open(file) as f:
                if file.suffix == ".jsonl":
                    for line in f:
                        data.append(json.loads(line))
                else:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)

        if self.limit:
            data = data[: self.limit]

        self._data = data

    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all samples from the task."""
        return self._data if self._data else []

    def __len__(self):
        """Get number of samples."""
        return len(self._data) if self._data else 0

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.get_samples())

