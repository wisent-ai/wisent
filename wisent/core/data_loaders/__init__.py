"""
Data loaders for various benchmarks.

This module provides data loaders for tasks that need special handling.
"""

from typing import Any, Dict, List, Optional


__all__ = ["LiveCodeBenchLoader"]


class LiveCodeBenchLoader:
    """
    Minimal stub for LiveCodeBench data loader.

    This is a placeholder to allow imports to work.
    The actual LiveCodeBench loading is handled by the task's fallback logic.
    """

    def list_available_versions(self) -> List[str]:
        """List available LiveCodeBench versions."""
        return ["release_v1", "release_v2"]

    def get_version_info(self, version: str) -> Dict[str, Any]:
        """Get information about a specific version."""
        return {
            "version": version,
            "description": f"LiveCodeBench {version}",
            "contest_start": "2023-01-01",
            "contest_end": "2024-12-31",
        }

    def load_problems(self, release_version: str, limit: Optional[int] = None) -> List[Any]:
        """
        Load LiveCodeBench problems.

        Note: This is a stub. The actual task uses fallback data.
        """
        raise NotImplementedError(
            "LiveCodeBench loading not fully implemented. "
            "The task will use fallback sample data."
        )
