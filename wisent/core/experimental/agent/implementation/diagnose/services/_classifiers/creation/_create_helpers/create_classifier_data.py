"""Data generation and benchmark relevance methods for ClassifierCreator."""

from typing import Any, Dict, List

from wisent.core.constants import AGENT_DEFAULT_TIME_BUDGET, BENCHMARK_LOADING_TIME_DEFAULT, TASK_MIN_RELEVANCE_SCORE, DISPLAY_TOP_N_TINY
from ....model import Model


class DataGenerationMixin:
    """Mixin providing training data generation and benchmark discovery methods."""

    model: Model

    def _generate_training_data(self, issue_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generate training data dynamically for a specific issue type using relevant benchmarks.

        Args:
            issue_type: Type of issue to generate data for
            num_samples: Number of training samples to generate

        Returns:
            List of training examples with harmful/harmless pairs
        """
        print(f"   Loading dynamic training data for {issue_type}...")

        # Try to find relevant benchmarks for the issue type (using default 5-minute budget)
        relevant_benchmarks = self._find_relevant_benchmarks(issue_type)

        if relevant_benchmarks:
            print(f"   Found {len(relevant_benchmarks)} relevant benchmarks: {relevant_benchmarks[:DISPLAY_TOP_N_TINY]}...")
            return self._load_benchmark_data(relevant_benchmarks, num_samples)
        print("   No specific benchmarks found, using synthetic generation...")
        return self._generate_synthetic_training_data(issue_type, num_samples)

    def _find_relevant_benchmarks(self, issue_type: str, time_budget_minutes: float = AGENT_DEFAULT_TIME_BUDGET) -> List[str]:
        """Find relevant benchmarks for the given issue type based on time budget with priority-aware selection."""
        from ...budget import calculate_max_tasks_for_time_budget
        from ..tasks.task_relevance import find_relevant_tasks

        try:
            # Calculate max tasks using budget system
            max_tasks = calculate_max_tasks_for_time_budget(
                task_type="benchmark_evaluation", time_budget_minutes=time_budget_minutes
            )

            print(f"   Time budget: {time_budget_minutes:.1f}min -> max {max_tasks} tasks")

            # Use priority-aware intelligent benchmark selection
            try:
                # Import priority-aware selection function
                import os
                import sys

                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lm-harness-integration"))
                from only_benchmarks import find_most_relevant_benchmarks

                # Use priority-aware selection with time budget
                relevant_results = find_most_relevant_benchmarks(
                    prompt=issue_type,
                    top_k=max_tasks,
                    priority="all",
                    fast_only=False,
                    time_budget_minutes=time_budget_minutes,
                    prefer_fast=True,  # Prefer fast benchmarks for agent use
                )

                # Extract benchmark names
                relevant_benchmarks = [result["benchmark"] for result in relevant_results]

                if relevant_benchmarks:
                    print(f"   Found {len(relevant_benchmarks)} priority-aware benchmarks for '{issue_type}':")
                    for i, result in enumerate(relevant_results[:DISPLAY_TOP_N_TINY]):
                        priority_str = f" (priority: {result.get('priority', 'unknown')})"
                        loading_time_str = f" (loading time: {result.get('loading_time', BENCHMARK_LOADING_TIME_DEFAULT):.1f}s)"
                        print(f"      {i + 1}. {result['benchmark']}{priority_str}{loading_time_str}")
                    if len(relevant_benchmarks) > DISPLAY_TOP_N_TINY:
                        print(f"      ... and {len(relevant_benchmarks) - DISPLAY_TOP_N_TINY} more")

                return relevant_benchmarks

            except Exception as priority_error:
                print(f"   Priority-aware selection failed: {priority_error}")
                print("   Using legacy task relevance instead...")

                # Use legacy system
                relevant_task_results = find_relevant_tasks(
                    query=issue_type, max_results=max_tasks, min_relevance_score=TASK_MIN_RELEVANCE_SCORE
                )

                # Extract just the task names
                candidate_benchmarks = [task_name for task_name, score in relevant_task_results]

                # Use priority-aware budget optimization
                from ...budget import optimize_benchmarks_for_budget

                relevant_benchmarks = optimize_benchmarks_for_budget(
                    task_candidates=candidate_benchmarks,
                    time_budget_minutes=time_budget_minutes,
                    max_tasks=max_tasks,
                    prefer_fast=True,  # Agent prefers fast benchmarks
                )

                if relevant_benchmarks:
                    print(f"   Found {len(relevant_benchmarks)} relevant benchmarks for '{issue_type}':")
                    # Show the scores for the selected benchmarks
                    for i, (task_name, score) in enumerate(relevant_task_results[:DISPLAY_TOP_N_TINY]):
                        if task_name in relevant_benchmarks:
                            print(f"      {i + 1}. {task_name} (relevance: {score:.3f})")
                    if len(relevant_benchmarks) > DISPLAY_TOP_N_TINY:
                        print(f"      ... and {len(relevant_benchmarks) - DISPLAY_TOP_N_TINY} more")

                return relevant_benchmarks

        except Exception as e:
            print(f"   Error finding relevant benchmarks: {e}")
            print("   Using default tasks")
            # Minimal set of high priority fast benchmarks
            return ["mmlu", "truthfulqa_mc1", "hellaswag"]

    def _extract_benchmark_concepts(self, benchmark_names: List[str]) -> Dict[str, List[str]]:
        """Extract semantic concepts from benchmark names."""
        concepts = {}

        for name in benchmark_names:
            # Extract concepts from benchmark name
            name_concepts = []
            name_lower = name.lower()

            # Split on common separators and extract meaningful tokens
            tokens = name_lower.replace("_", " ").replace("-", " ").split()

            # Filter out common non-semantic tokens
            semantic_tokens = []
            skip_tokens = {
                "the",
                "and",
                "or",
                "of",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
                "from",
                "as",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "light",
                "full",
                "val",
                "test",
                "dev",
                "mc1",
                "mc2",
                "mt",
                "cot",
                "fewshot",
                "zeroshot",
                "generate",
                "until",
                "multiple",
                "choice",
                "group",
                "subset",
            }

            for token in tokens:
                if len(token) > 2 and token not in skip_tokens and token.isalpha():
                    semantic_tokens.append(token)

            # Extract domain-specific concepts
            domain_concepts = self._extract_domain_concepts(name_lower, semantic_tokens)
            name_concepts.extend(domain_concepts)

            concepts[name] = list(set(name_concepts))  # Remove duplicates

        return concepts

    def _extract_domain_concepts(self, benchmark_name: str, tokens: List[str]) -> List[str]:
        """Extract domain-specific concepts directly from benchmark name components."""
        concepts = []

        # Add all meaningful tokens as concepts
        for token in tokens:
            if len(token) > 2:
                concepts.append(token)

        # Extract compound concept meanings from token combinations
        name_parts = benchmark_name.lower().split("_")

        # Generate concept combinations
        for i, part in enumerate(name_parts):
            if len(part) > 2:
                concepts.append(part)

                # Look for meaningful compound concepts
                if i < len(name_parts) - 1:
                    next_part = name_parts[i + 1]
                    if len(next_part) > 2:
                        compound = f"{part}_{next_part}"
                        concepts.append(compound)

        # Extract semantic root words
        for token in tokens:
            root_concepts = self._extract_semantic_roots(token)
            concepts.extend(root_concepts)

        return list(set(concepts))  # Remove duplicates

    def _extract_semantic_roots(self, word: str) -> List[str]:
        """Extract semantic root concepts from a word."""
        roots = []

        # Simple morphological analysis
        # Remove common suffixes to find roots
        suffixes = [
            "ing",
            "tion",
            "sion",
            "ness",
            "ment",
            "able",
            "ible",
            "ful",
            "less",
            "ly",
            "al",
            "ic",
            "ous",
            "ive",
        ]

        root = word
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[: -len(suffix)]
                break

        if root != word and len(root) > 2:
            roots.append(root)

        # Add the original word
        roots.append(word)

        return roots
