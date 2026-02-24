"""Maps Group E threshold constants to their metric collection functions.

Each entry specifies:
- constant_name: which threshold to calibrate
- collector: callable(model, pairs, task_name) -> List[float] of observations
- default_quantile: percentile at which to set the threshold
- description: what the metric measures
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from wisent.core.constants import (
    CALIBRATION_Q_NLI_MARGIN, CALIBRATION_Q_NLI_ENT, CALIBRATION_Q_EMB_DELTA,
    CALIBRATION_Q_EMB_MATCH, CALIBRATION_Q_PROBE_HIGH, CALIBRATION_Q_PROBE_LOW,
)


@dataclass(frozen=True)
class ThresholdMetricSpec:
    """Specification for how to calibrate a single threshold constant."""

    constant_name: str
    collector_name: str
    default_quantile: float
    description: str
    invert: bool = False


def _collect_nli_margins(model: Any, pairs: Any, task_name: str) -> List[float]:
    """Collect NLI entailment margins from pair evaluations."""
    from wisent.core.evaluation.evaluators.nli_evaluator import NLIEvaluator
    evaluator = NLIEvaluator()
    margins = []
    for pair in pairs:
        result = evaluator.evaluate_pair(pair, task_name=task_name)
        if "margin" in result:
            margins.append(float(result["margin"]))
    return margins


def _collect_nli_entailment(model: Any, pairs: Any, task_name: str) -> List[float]:
    """Collect NLI entailment scores from pair evaluations."""
    from wisent.core.evaluation.evaluators.nli_evaluator import NLIEvaluator
    evaluator = NLIEvaluator()
    scores = []
    for pair in pairs:
        result = evaluator.evaluate_pair(pair, task_name=task_name)
        if "entailment" in result:
            scores.append(float(result["entailment"]))
    return scores


def _collect_embedding_deltas(model: Any, pairs: Any, task_name: str) -> List[float]:
    """Collect embedding cosine deltas between positive and negative."""
    from wisent.core.evaluation.evaluators.embedding_evaluator import (
        EmbeddingEvaluator,
    )
    evaluator = EmbeddingEvaluator()
    deltas = []
    for pair in pairs:
        result = evaluator.evaluate_pair(pair, task_name=task_name)
        if "delta" in result:
            deltas.append(float(result["delta"]))
    return deltas


def _collect_embedding_matches(model: Any, pairs: Any, task_name: str) -> List[float]:
    """Collect embedding match scores."""
    from wisent.core.evaluation.evaluators.embedding_evaluator import (
        EmbeddingEvaluator,
    )
    evaluator = EmbeddingEvaluator()
    matches = []
    for pair in pairs:
        result = evaluator.evaluate_pair(pair, task_name=task_name)
        if "match_score" in result:
            matches.append(float(result["match_score"]))
    return matches


def _collect_linear_probe_accuracies(
    model: Any, pairs: Any, task_name: str,
) -> List[float]:
    """Collect linear probe accuracies across multiple layer samples."""
    from wisent.core.geometry import compute_geometry_metrics
    import torch
    accuracies = []
    pos_acts = []
    neg_acts = []
    for pair in pairs:
        p_act = getattr(pair, 'positive_activation', None)
        n_act = getattr(pair, 'negative_activation', None)
        if p_act is not None and n_act is not None:
            pos_acts.append(p_act)
            neg_acts.append(n_act)
    if len(pos_acts) >= 5:
        pos_t = torch.stack(pos_acts)
        neg_t = torch.stack(neg_acts)
        metrics = compute_geometry_metrics(pos_t, neg_t)
        lpa = metrics.get("linear_probe_accuracy")
        if lpa is not None:
            accuracies.append(float(lpa))
    return accuracies


COLLECTOR_REGISTRY: Dict[str, Callable] = {
    "nli_margins": _collect_nli_margins,
    "nli_entailment": _collect_nli_entailment,
    "embedding_deltas": _collect_embedding_deltas,
    "embedding_matches": _collect_embedding_matches,
    "linear_probe_accuracies": _collect_linear_probe_accuracies,
}


THRESHOLD_METRIC_MAP: List[ThresholdMetricSpec] = [
    ThresholdMetricSpec(
        constant_name="NLI_MARGIN",
        collector_name="nli_margins",
        default_quantile=CALIBRATION_Q_NLI_MARGIN,
        description="NLI margin distribution; low quantile marks minimum",
    ),
    ThresholdMetricSpec(
        constant_name="NLI_ENT_MIN",
        collector_name="nli_entailment",
        default_quantile=CALIBRATION_Q_NLI_ENT,
        description="NLI entailment score distribution; low quantile marks minimum",
    ),
    ThresholdMetricSpec(
        constant_name="EMB_DELTA_MIN",
        collector_name="embedding_deltas",
        default_quantile=CALIBRATION_Q_EMB_DELTA,
        description="Embedding delta distribution; low quantile marks minimum",
    ),
    ThresholdMetricSpec(
        constant_name="EMB_MATCH_MIN",
        collector_name="embedding_matches",
        default_quantile=CALIBRATION_Q_EMB_MATCH,
        description="Embedding match distribution; low quantile marks minimum",
    ),
    ThresholdMetricSpec(
        constant_name="RECOMMEND_LINEAR_PROBE_HIGH",
        collector_name="linear_probe_accuracies",
        default_quantile=CALIBRATION_Q_PROBE_HIGH,
        description="Linear probe accuracy distribution; high quantile marks high",
    ),
    ThresholdMetricSpec(
        constant_name="RECOMMEND_LINEAR_PROBE_LOW",
        collector_name="linear_probe_accuracies",
        default_quantile=CALIBRATION_Q_PROBE_LOW,
        description="Linear probe accuracy distribution; low quantile marks low",
    ),
]


def get_collector(name: str) -> Optional[Callable]:
    """Get a metric collector function by name."""
    return COLLECTOR_REGISTRY.get(name)
