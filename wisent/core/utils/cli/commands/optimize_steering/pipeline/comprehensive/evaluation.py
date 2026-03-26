"""Baseline evaluation for comprehensive optimization — delegates to HF cache."""
from __future__ import annotations

import logging
from typing import Optional

from wisent.core.utils.config_tools.constants import SCORE_RANGE_MIN

from wisent.core.utils.cli.optimize_steering.pipeline.comprehensive import baseline_cache

logger = logging.getLogger(__name__)


def compute_baseline_score(
    model: str,
    task_name: str,
    pairs_file: str,
    device: Optional[str],
    verbose: bool,
) -> float:
    """Compute unsteered model accuracy, using HF cache when available.

    Checks HuggingFace for a cached baseline first. If found, loads the
    cached accuracy. Otherwise generates baseline responses, evaluates,
    uploads to HF for future reuse, and returns the accuracy.
    """
    if baseline_cache.check_baseline_exists(model, task_name):
        if verbose:
            logger.info("Loading cached baseline from HuggingFace...")
        _, scores, meta = baseline_cache.load_baseline_from_hf(
            model, task_name,
        )
        accuracy = meta.get("accuracy")
        if accuracy is not None:
            return accuracy
        if scores:
            return sum(s["correct"] for s in scores) / len(scores)
        return SCORE_RANGE_MIN

    if verbose:
        logger.info("Generating baseline (no HF cache found)...")
    accuracy, _, _ = baseline_cache.generate_and_upload_baseline(
        model, task_name, pairs_file, device,
    )
    return accuracy
