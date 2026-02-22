"""
Data loaders for various benchmarks.

This module provides data loaders for tasks that need special handling.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


__all__ = ["LiveCodeBenchLoader", "LiveCodeBenchProblem"]


@dataclass
class LiveCodeBenchProblem:
    """
    A LiveCodeBench coding problem.

    Attributes:
        question_title: Title of the problem
        question_content: Full problem description
        platform: Source platform (codeforces, leetcode, atcoder)
        question_id: Unique problem identifier
        contest_id: Contest identifier
        contest_date: Date of the contest
        starter_code: Optional starter code template
        difficulty: Problem difficulty (easy, medium, hard)
        public_test_cases: Public test cases
        private_test_cases: Private test cases
        metadata: Additional metadata
        answer: Correct answer/solution (for TaskInterface compatibility)
        good_code: Code that passes tests (from wisent-core)
        bad_code: Code that fails tests (from wisent-core)
    """
    question_title: str
    question_content: str
    platform: str
    question_id: str
    contest_id: str
    contest_date: str
    starter_code: str
    difficulty: str
    public_test_cases: List[Any]
    private_test_cases: List[Any]
    metadata: Dict[str, Any]
    answer: Optional[str] = None
    good_code: Optional[str] = None
    bad_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task_id": self.question_id,
            "question_id": self.question_id,  # Also include question_id for compatibility
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.upper(),
            "contest_id": self.contest_id,
            "contest_date": self.contest_date,
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.upper(),
            "public_test_cases": [
                {
                    "input": tc if isinstance(tc, str) else str(tc),
                    "output": "",
                    "testtype": "FUNCTIONAL"
                }
                for tc in self.public_test_cases[:3]  # Limit to first 3 for brevity
            ] if self.public_test_cases else [],
            "metadata": self.metadata,
            "answer": self.good_code,  # Use good_code as the correct answer
            "good_code": self.good_code,
            "bad_code": self.bad_code,
        }


class LiveCodeBenchLoader:
    """
    LiveCodeBench data loader.

    Loads real coding problems from the LiveCodeBench dataset on HuggingFace.
    Dataset: livecodebench/code_generation_lite

    Also loads pre-generated good/bad code solutions from local cache.
    """

    def __init__(self, solution_cache_dir: Optional[str] = None):
        self._dataset_name = "livecodebench/code_generation_lite"
        self._cached_dataset = None
        self._solution_data = None
        self.solution_cache_dir = solution_cache_dir or "./livecodebench_solutions"

    def list_available_versions(self) -> List[str]:
        """List available LiveCodeBench versions."""
        # The dataset doesn't have explicit versions, but we can filter by date ranges
        return ["release_v1", "release_v2", "all"]

    def get_version_info(self, version: str) -> Dict[str, Any]:
        """Get information about a specific version."""
        version_info = {
            "release_v1": {
                "version": "release_v1",
                "description": "LiveCodeBench Release V1 (May 2023 - Oct 2023)",
                "problems": "~500",
                "date_range": "2023-05-01 to 2023-10-31",
            },
            "release_v2": {
                "version": "release_v2",
                "description": "LiveCodeBench Release V2 (Nov 2023 - Apr 2024)",
                "problems": "~500",
                "date_range": "2023-11-01 to 2024-04-30",
            },
            "all": {
                "version": "all",
                "description": "All LiveCodeBench problems",
                "problems": "1055",
                "date_range": "2023-05-01 to 2024-12-31",
            }
        }
        return version_info.get(version, version_info["all"])

    def _load_solution_data(self) -> Dict[str, Any]:
        """
        Load pre-generated AI model solutions from local cache.

        Returns:
            Dictionary with question_id -> {good_code, bad_code, difficulty} mapping.
        """
        import json
        from pathlib import Path

        if self._solution_data is not None:
            return self._solution_data

        cache_file = Path(self.solution_cache_dir) / "solutions.json"

        if not cache_file.exists():
            import logging
            logging.warning(
                f"Solutions cache not found at {cache_file}. "
                f"Run solution generation first using LiveCodeBenchSolutionGenerator. "
                f"Problems will have no answer field."
            )
            self._solution_data = {}
            return {}

        with open(cache_file, 'r') as f:
            data = json.load(f)

        # Create mapping from question_id to solutions
        solution_map = {}
        for problem in data.get("problems", []):
            question_id = problem.get("question_id")
            if question_id and problem.get("good_example") and problem.get("bad_example"):
                solution_map[question_id] = {
                    "good_code": problem["good_example"].get("code", ""),
                    "bad_code": problem["bad_example"].get("code", ""),
                    "difficulty": problem.get("difficulty", "unknown"),
                }

        self._solution_data = solution_map
        return solution_map

    def load_problems(
        self,
        release_version: str = "all",
        limit: Optional[int] = None
    ) -> List[LiveCodeBenchProblem]:
        """
        Load LiveCodeBench problems from HuggingFace.

        Arguments:
            release_version: Version to load (release_v1, release_v2, or all)
            limit: Maximum number of problems to load

        Returns:
            List of LiveCodeBenchProblem objects
        """
        from datasets import load_dataset

        # Load dataset (cached after first load)
        if self._cached_dataset is None:
            self._cached_dataset = load_dataset(self._dataset_name, split="test")

        dataset = self._cached_dataset

        # Filter by version if needed
        if release_version == "release_v1":
            # Filter problems from May 2023 - Oct 2023
            dataset = dataset.filter(
                lambda x: x["contest_date"] >= "2023-05-01" and x["contest_date"] <= "2023-10-31"
            )
        elif release_version == "release_v2":
            # Filter problems from Nov 2023 - Apr 2024
            dataset = dataset.filter(
                lambda x: x["contest_date"] >= "2023-11-01" and x["contest_date"] <= "2024-04-30"
            )
        # "all" or any other value: use all problems

        # Apply limit
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        # Load solution data from wisent-core
        solution_map = self._load_solution_data()

        # Convert to LiveCodeBenchProblem objects
        problems = []
        for item in dataset:
            question_id = item.get("question_id", "")

            # Get solutions if available
            solutions = solution_map.get(question_id, {})
            good_code = solutions.get("good_code")
            bad_code = solutions.get("bad_code")

            problem = LiveCodeBenchProblem(
                question_title=item.get("question_title", ""),
                question_content=item.get("question_content", ""),
                platform=item.get("platform", ""),
                question_id=question_id,
                contest_id=str(item.get("contest_id", "")),
                contest_date=item.get("contest_date", ""),
                starter_code=item.get("starter_code", ""),
                difficulty=item.get("difficulty", ""),
                public_test_cases=item.get("public_test_cases", []),
                private_test_cases=item.get("private_test_cases", []),
                metadata=item.get("metadata", {}),
                answer=good_code,  # Set answer field for TaskInterface
                good_code=good_code,
                bad_code=bad_code,
            )
            problems.append(problem)

        return problems
