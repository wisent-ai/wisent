"""Shared dataclasses and helpers for contrastive pair diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(slots=True)
class DiagnosticsConfig:
    """Threshold configuration for diagnostics.

    Attributes:
        warn_on_missing_activations: Whether missing activations should be reported as issues.
        min_unique_prompt_ratio: Minimum ratio of unique prompts.
        min_average_length: Minimum average response length.
        max_duplicate_fraction: Maximum allowed fraction of exact duplicates.
        dedup_item_threshold: Threshold below which to use exact-match dedup.
    """

    warn_on_missing_activations: bool = True
    min_unique_prompt_ratio: float = None
    min_average_length: int = None
    max_duplicate_fraction: float = None
    dedup_item_threshold: int = None


@dataclass(slots=True)
class DiagnosticsIssue:
    """Represents a single diagnostics issue detected in a pair set."""

    metric: str
    severity: str
    message: str
    pair_index: int | None = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricReport:
    """Stores summary statistics for a single diagnostics metric."""

    name: str
    summary: Dict[str, Any]
    issues: List[DiagnosticsIssue] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticsReport:
    """Aggregated diagnostics results across metrics."""

    metrics: Dict[str, MetricReport]
    issues: List[DiagnosticsIssue]
    summary: Dict[str, Any]
    has_critical_issues: bool

    @classmethod
    def from_metrics(cls, reports: Iterable[MetricReport]) -> "DiagnosticsReport":
        metrics_map: Dict[str, MetricReport] = {}
        all_issues: List[DiagnosticsIssue] = []

        for report in reports:
            metrics_map[report.name] = report
            all_issues.extend(report.issues)

        summary = {name: report.summary for name, report in metrics_map.items()}
        has_critical = any(issue.severity == "critical" for issue in all_issues)

        return cls(metrics=metrics_map, issues=all_issues, summary=summary, has_critical_issues=has_critical)
