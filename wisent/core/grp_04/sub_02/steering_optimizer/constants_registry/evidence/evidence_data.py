"""Data models for the evidence ledger system.

AxisEvidence records empirical results from comparing values of a
single search-space axis.  AxisReduction is the actionable output
consumed by the search-space generator to prune dead values.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Values within this margin of the best score count as dominant
DOMINANCE_MARGIN: float = 0.02

# Evidence older than this is considered stale
MAX_AGE_DAYS: int = 90

# Confidence multiplier when applying evidence from a sibling model
CROSS_MODEL_CONFIDENCE_DECAY: float = 0.5


def _model_family(model_name: str) -> str:
    """Derive a coarse model-family string from a full model name.

    Examples::

        "meta-llama/Llama-3.2-1B-Instruct" -> "llama-3.2"
        "Qwen/Qwen3-8B"                    -> "qwen3"
    """
    base = model_name.rsplit("/", 1)[-1].lower()
    for suffix in ("-instruct", "-chat", "-base", "-hf"):
        base = base.replace(suffix, "")
    parts = base.split("-")
    family_parts: list[str] = []
    for p in parts:
        stripped = p.rstrip("bm")
        if stripped.replace(".", "").isdigit() and len(family_parts) > 0:
            break
        family_parts.append(p)
    return "-".join(family_parts) if family_parts else base


def compute_dominant_values(
    scores: Dict[str, float],
    margin: float = DOMINANCE_MARGIN,
) -> Tuple[List[str], float]:
    """Return (dominant_values, margin_gap) from a scores dict."""
    if not scores:
        return [], 0.0
    best_val = max(scores.values())
    dominant = [k for k, v in scores.items() if best_val - v <= margin]
    sorted_vals = sorted(scores.values(), reverse=True)
    gap = (sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else 0.0
    return dominant, gap


@dataclass
class AxisEvidence:
    """One experiment record: scores for each value of a single axis."""

    axis_name: str
    model_name: str
    task_name: str
    method_name: str
    tested_values: List[str]
    scores: Dict[str, float]
    dominant_values: List[str]
    margin: float
    confidence: float
    n_samples: int
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(),
    )
    source: str = "compare_axis"
    notes: str = ""

    @property
    def model_family(self) -> str:
        return _model_family(self.model_name)

    @property
    def id(self) -> str:
        """Deterministic short id for this record."""
        blob = (
            f"{self.axis_name}:{self.model_name}:"
            f"{self.task_name}:{self.method_name}:"
            f"{self.created_at}"
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "axis_name": self.axis_name,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "task_name": self.task_name,
            "method_name": self.method_name,
            "tested_values": self.tested_values,
            "scores": self.scores,
            "dominant_values": self.dominant_values,
            "margin": self.margin,
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "created_at": self.created_at,
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AxisEvidence:
        return cls(
            axis_name=data["axis_name"],
            model_name=data["model_name"],
            task_name=data["task_name"],
            method_name=data["method_name"],
            tested_values=data["tested_values"],
            scores=data["scores"],
            dominant_values=data["dominant_values"],
            margin=data["margin"],
            confidence=data["confidence"],
            n_samples=data["n_samples"],
            created_at=data.get("created_at", ""),
            source=data.get("source", "unknown"),
            notes=data.get("notes", ""),
        )


@dataclass
class AxisReduction:
    """Actionable output: which values to keep or remove for one axis."""

    axis_name: str
    keep_values: List[str]
    removed_values: List[str]
    evidence_ids: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis_name": self.axis_name,
            "keep_values": self.keep_values,
            "removed_values": self.removed_values,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AxisReduction:
        return cls(
            axis_name=data["axis_name"],
            keep_values=data["keep_values"],
            removed_values=data["removed_values"],
            evidence_ids=data.get("evidence_ids", []),
            confidence=data.get("confidence", 0.0),
        )
