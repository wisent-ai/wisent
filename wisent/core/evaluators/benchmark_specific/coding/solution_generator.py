"""
LiveCodeBench solution generator using AI models and code execution.

This module generates and evaluates solutions for LiveCodeBench problems,
creating good/bad code pairs for contrastive learning.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, asdict

from wisent.core.evaluators.benchmark_specific.coding.providers.livecodebench.provider import LiveCodeBenchProvider
from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig
from wisent.core.evaluators.benchmark_specific.coding.providers.core.atoms import CodingTask


@dataclass
class SolutionExample:
    """A single solution example with evaluation result."""
    model: str
    code: str
    result: str  # "good" or "bad"
    status: str  # "ok", "compile_error", "runtime_error", "timeout"
    elapsed: float


@dataclass
class ProblemSolutions:
    """Solutions for a single problem."""
    question_id: str
    good_example: Optional[dict[str, Any]] = None
    bad_example: Optional[dict[str, Any]] = None
    difficulty: str = "unknown"
    all_solutions: list[dict[str, Any]] = None

    def __post_init__(self):
        if self.all_solutions is None:
            self.all_solutions = []


class LiveCodeBenchSolutionGenerator:
    """
    Generates and evaluates solutions for LiveCodeBench problems.

    This replicates the wisent-core approach but as an independent system.
    """

    def __init__(
        self,
        model_fns: dict[str, Callable[[CodingTask], dict[str, str]]],
        cache_dir: str = "./livecodebench_solutions",
        evaluator_config: Optional[EvaluatorConfig] = None,
    ):
        """
        Initialize the solution generator.

        Args:
            model_fns: Dictionary mapping model names to solution generation functions.
                      Each function takes a CodingTask and returns a dict of files.
            cache_dir: Directory to cache generated solutions.
            evaluator_config: Optional configuration for code evaluation.
        """
        self.model_fns = model_fns
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator_config = evaluator_config or EvaluatorConfig(
            image="coding/sandbox:polyglot-1.0",
            self_repair=False,
            time_limit_s=8,
            cpu_limit_s=3,
            mem_limit_mb=768,
        )

        self.cache_file = self.cache_dir / "solutions.json"
        self._cached_solutions: Optional[dict[str, ProblemSolutions]] = None

    def _load_cache(self) -> dict[str, ProblemSolutions]:
        """Load cached solutions from disk."""
        if self._cached_solutions is not None:
            return self._cached_solutions

        if not self.cache_file.exists():
            self._cached_solutions = {}
            return {}

        with open(self.cache_file, 'r') as f:
            data = json.load(f)

        solutions_map = {}
        for item in data.get("problems", []):
            problem_id = item["question_id"]
            solutions_map[problem_id] = ProblemSolutions(
                question_id=problem_id,
                good_example=item.get("good_example"),
                bad_example=item.get("bad_example"),
                difficulty=item.get("difficulty", "unknown"),
                all_solutions=item.get("all_solutions", []),
            )

        self._cached_solutions = solutions_map
        return solutions_map

    def _save_cache(self, solutions: dict[str, ProblemSolutions]):
        """Save solutions to disk."""
        data = {
            "total_problems": len(solutions),
            "problems": [
                {
                    "question_id": ps.question_id,
                    "good_example": ps.good_example,
                    "bad_example": ps.bad_example,
                    "difficulty": ps.difficulty,
                    "all_solutions": ps.all_solutions,
                }
                for ps in solutions.values()
            ]
        }

        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(solutions)} problem solutions to {self.cache_file}")

    def generate_solutions(
        self,
        limit: Optional[int] = None,
        platform: Optional[str] = None,
        release_version: str = "all",
        skip_existing: bool = True,
    ):
        """
        Generate solutions for LiveCodeBench problems using multiple AI models.

        Args:
            limit: Maximum number of problems to process.
            platform: Filter by platform (leetcode, codeforces, atcoder).
            release_version: Dataset version (release_v1, release_v2, all).
            skip_existing: Skip problems that already have good/bad pairs.
        """
        # Load cache
        cached_solutions = self._load_cache()

        # Load problems
        provider = LiveCodeBenchProvider(
            language="python",
            limit=limit,
            platform=platform,
            release_version=release_version,
        )

        problems_processed = 0
        problems_skipped = 0

        print(f"Processing LiveCodeBench problems...")
        print(f"Models: {list(self.model_fns.keys())}")

        for idx, task in enumerate(provider.iter_tasks()):
            question_id = task.options.get("problem_id", f"unknown_{idx}")

            # Skip if already has good/bad pair
            if skip_existing and question_id in cached_solutions:
                existing = cached_solutions[question_id]
                if existing.good_example and existing.bad_example:
                    problems_skipped += 1
                    continue

            print(f"\n[{idx + 1}] Processing {question_id}...")

            # Generate solutions with each model
            solutions = []
            for model_name, model_fn in self.model_fns.items():
                print(f"  - Generating with {model_name}...")

                try:
                    # Generate solution
                    files = model_fn(task)

                    # Evaluate solution
                    evaluator = CodingEvaluator(
                        provider=None,  # Not used for single evaluation
                        model_fn=lambda _: files,
                        repair_fn=None,
                        cfg=self.evaluator_config,
                    )

                    result = evaluator._run_once(task, {**task.files, **files})

                    # Determine if good or bad
                    is_good = result.status == "ok"

                    solution = SolutionExample(
                        model=model_name,
                        code=files.get("solution.py", ""),
                        result="good" if is_good else "bad",
                        status=result.status,
                        elapsed=result.elapsed,
                    )
                    solutions.append(solution)

                    print(f"    Result: {solution.result} ({solution.status}, {solution.elapsed:.2f}s)")

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            # Select best good and bad examples
            good_solutions = [s for s in solutions if s.result == "good"]
            bad_solutions = [s for s in solutions if s.result == "bad"]

            problem_solution = ProblemSolutions(
                question_id=question_id,
                difficulty=task.options.get("difficulty", "unknown"),
                all_solutions=[asdict(s) for s in solutions],
            )

            if good_solutions:
                # Prefer fastest good solution
                best_good = min(good_solutions, key=lambda s: s.elapsed)
                problem_solution.good_example = asdict(best_good)

            if bad_solutions:
                # Prefer bad solution with fastest failure
                best_bad = min(bad_solutions, key=lambda s: s.elapsed)
                problem_solution.bad_example = asdict(best_bad)

            # Update cache
            cached_solutions[question_id] = problem_solution
            problems_processed += 1

            # Save periodically
            if problems_processed % 10 == 0:
                self._save_cache(cached_solutions)
                print(f"\nProgress: {problems_processed} processed, {problems_skipped} skipped")

        # Final save
        self._save_cache(cached_solutions)

        print(f"\n=== Generation Complete ===")
        print(f"Problems processed: {problems_processed}")
        print(f"Problems skipped: {problems_skipped}")
        print(f"Total in cache: {len(cached_solutions)}")

        # Summary statistics
        with_good_bad = sum(1 for ps in cached_solutions.values() if ps.good_example and ps.bad_example)
        print(f"Problems with good+bad pairs: {with_good_bad}")

    def get_solutions(self, question_id: str) -> Optional[ProblemSolutions]:
        """Get solutions for a specific problem."""
        cached = self._load_cache()
        return cached.get(question_id)

    def get_all_solutions(self) -> dict[str, ProblemSolutions]:
        """Get all cached solutions."""
        return self._load_cache()
