"""
Benchmark processing for lm-eval-harness tasks.

This package provides comprehensive benchmark processing functionality including:
- Benchmark registry with priority-based task configurations
- Sample extraction from lm-eval-harness tasks
- README parsing for automatic tag generation
- Benchmark matching using LLM analysis
- Priority-based filtering for agentic optimization
"""

from __future__ import annotations

# Constants
from .constants import (
    LM_EVAL_TASKS_PATH,
    APPROVED_SKILLS,
    APPROVED_RISKS,
)

# Registry
from .registry import CORE_BENCHMARKS, BENCHMARKS

# Sample extraction
from .sample_extraction import (
    get_task_samples_for_analysis,
    get_task_samples_with_subtasks,
)
from .sample_helpers import (
    try_alternative_task_names,
    get_task_samples_direct,
    try_datasets_direct_load,
)

# README parsing
from .readme_parsing import (
    extract_readme_info,
    determine_skill_risk_tags,
    update_benchmark_from_readme,
    update_all_benchmarks_from_readme,
)

# Matching and filtering
from .matching import (
    BENCHMARK_DESCRIPTIONS,
    apply_priority_filtering,
    get_benchmarks_by_priority,
    get_priority_summary,
    print_priority_summary,
    find_most_relevant_benchmarks,
)

# Testing
from .testing import (
    test_single_benchmark_direct,
    test_benchmark_creation,
    extract_contrastive_pairs_from_output,
    test_readme_updates,
    test_benchmark_matching,
    main,
    cli_entry,
)


__all__ = [
    # Constants
    "LM_EVAL_TASKS_PATH",
    "APPROVED_SKILLS",
    "APPROVED_RISKS",
    # Registry
    "CORE_BENCHMARKS",
    "BENCHMARKS",
    # Sample extraction
    "get_task_samples_for_analysis",
    "get_task_samples_with_subtasks",
    "try_alternative_task_names",
    "get_task_samples_direct",
    "try_datasets_direct_load",
    # README parsing
    "extract_readme_info",
    "determine_skill_risk_tags",
    "update_benchmark_from_readme",
    "update_all_benchmarks_from_readme",
    # Matching and filtering
    "BENCHMARK_DESCRIPTIONS",
    "apply_priority_filtering",
    "get_benchmarks_by_priority",
    "get_priority_summary",
    "print_priority_summary",
    "find_most_relevant_benchmarks",
    # Testing
    "test_single_benchmark_direct",
    "test_benchmark_creation",
    "extract_contrastive_pairs_from_output",
    "test_readme_updates",
    "test_benchmark_matching",
    "main",
    "cli_entry",
]
