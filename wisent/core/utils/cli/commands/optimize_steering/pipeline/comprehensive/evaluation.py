"""Baseline evaluation and result helpers for comprehensive optimization."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, RECURSION_INITIAL_DEPTH, COMBO_OFFSET,
)

logger = logging.getLogger(__name__)


def compute_baseline_score(
    model: str,
    task_name: str,
    pairs_file: str,
    device: Optional[str],
    verbose: bool,
) -> float:
    """Compute unsteered model accuracy on evaluation pairs.

    Loads the model, generates responses without any steering, and
    evaluates them against ground truth using the task evaluator.
    """
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.models import get_generate_kwargs
    from wisent.core.reading.evaluators.rotator import EvaluatorRotator

    EvaluatorRotator.discover_evaluators(
        "wisent.core.reading.evaluators.oracles",
    )
    EvaluatorRotator.discover_evaluators(
        "wisent.core.reading.evaluators.benchmark_specific",
    )

    with open(pairs_file) as f:
        data = json.load(f)

    pairs_data = data.get("pairs", [])
    if not pairs_data:
        logger.warning("No pairs found in %s", pairs_file)
        return SCORE_RANGE_MIN

    wisent_model = WisentModel(model, device=device)
    evaluator = EvaluatorRotator(
        evaluator=None, task_name=task_name, autoload=False,
    )
    gen_kwargs = get_generate_kwargs()

    correct = RECURSION_INITIAL_DEPTH
    total = RECURSION_INITIAL_DEPTH

    for pair_dict in pairs_data:
        prompt = pair_dict.get("prompt", "")
        positive = pair_dict.get("positive_response", {})
        negative = pair_dict.get("negative_response", {})
        expected = positive.get("model_response", "")
        neg_response = negative.get("model_response", "")

        messages = [{"role": "user", "content": prompt}]
        response = wisent_model.generate([messages], **gen_kwargs)[
            RECURSION_INITIAL_DEPTH
        ]

        eval_kwargs = {
            "response": response,
            "expected": expected,
            "question": prompt,
            "choices": [neg_response, expected],
            "task_name": task_name,
        }
        metadata = pair_dict.get("metadata", {})
        if metadata:
            for key, value in metadata.items():
                if value is not None and key not in eval_kwargs:
                    eval_kwargs[key] = value

        result = evaluator.evaluate(**eval_kwargs)
        if result.ground_truth == "TRUTHFUL":
            correct += COMBO_OFFSET
        total += COMBO_OFFSET

    wisent_model.detach()

    if total == RECURSION_INITIAL_DEPTH:
        return SCORE_RANGE_MIN
    return correct / total
