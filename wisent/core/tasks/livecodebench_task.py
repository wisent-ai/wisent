"""
LiveCodeBench task implementation for task-agnostic architecture.
"""

from typing import Any, Dict, List, Optional

from ..benchmark_extractors import LiveCodeBenchExtractor
from ..data_loaders import LiveCodeBenchLoader
from ..task_interface import TaskInterface
from ..errors import VersionValidationError, VersionInfoError


class LiveCodeBenchTask(TaskInterface):
    """LiveCodeBench task implementation."""

    def __init__(self, release_version: str = "release_v1", limit: Optional[int] = None):
        self._extractor = LiveCodeBenchExtractor()
        self._data_loader = LiveCodeBenchLoader()
        self._release_version = release_version
        self._validate_release_version(release_version)
        self._data = None  # Cache for loaded data
        self._limit = limit  # Store limit for later use

    def _validate_release_version(self, release_version: str) -> None:
        """Validate release version."""
        try:
            valid_versions = set(self._data_loader.list_available_versions())
            if release_version not in valid_versions:
                raise VersionValidationError(version=release_version)
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # No fallback - raise error if version validation fails
            raise VersionValidationError(version=release_version, cause=e)

    def _get_version_info(self) -> Dict[str, Any]:
        """Get version-specific information."""
        try:
            return self._data_loader.get_version_info(self._release_version)
        except Exception as e:
            # No fallback - raise error if version info cannot be loaded
            raise VersionInfoError(version=self._release_version, cause=e)

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LiveCodeBench data for the specified release version."""
        # Load real LiveCodeBench data - no fallbacks
        problems = self._data_loader.load_problems(release_version=self._release_version, limit=limit)

        # Convert to dictionary format
        return [problem.to_dict() for problem in problems]

    def get_extractor(self):
        """Get the LiveCodeBench extractor."""
        return self._extractor

    def get_name(self) -> str:
        """Get the task name."""
        return "livecodebench"

    def get_description(self) -> str:
        """Get the task description."""
        version_info = self._get_version_info()
        return f"LiveCodeBench {self._release_version}: Contamination-free coding benchmark with {version_info['problems']} problems ({version_info['date_range']}) from LeetCode, AtCoder, and CodeForces"

    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["coding", "reasoning", "algorithms", "data-structures"]

    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # LiveCodeBench doesn't have separate validation sets

    def has_test_docs(self) -> bool:
        """Check if task has test documents."""
        return True  # All samples are considered test docs

    def test_docs(self) -> List[Dict[str, Any]]:
        """Get test documents."""
        if self._data is None:
            self._data = self.load_data(limit=self._limit)
        return self._data

    def validation_docs(self) -> List[Dict[str, Any]]:
        """Get validation documents."""
        return []  # No separate validation set

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text prompt."""
        # Combine problem description with starter code
        question = doc.get("question_content", "")
        starter = doc.get("starter_code", "")
        return f"{question}\n\n{starter}"
