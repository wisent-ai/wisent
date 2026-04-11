"""Per-response comparison for Benchmark Debug tab.

Loads baseline and winning trial responses from HF, compares
evaluations, and formats as a dataframe for display.
"""

import json
import logging
from typing import Optional

from wisent.core.utils.config_tools.constants import (
    DISPLAY_TRUNCATION_MEDIUM,
)

_log = logging.getLogger(__name__)


def _load_trial_scores_from_hf(
    model_name: str, task_name: str, method: str, trial_idx: int,
) -> Optional[list]:
    """Load a trial's scores (with evaluations) from HF."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
        model_to_safe_name,
    )
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _hf_hub_download,
    )
    safe = model_to_safe_name(model_name)
    hf_path = (
        f"trials/{safe}/{task_name}/{method.lower()}"
        f"/trial_{trial_idx:04d}/scores.json"
    )
    try:
        local = _hf_hub_download(hf_path)
        with open(local, "r") as f:
            data = json.load(f)
        return data.get("evaluations", [])
    except Exception as exc:
        _log.info(
            "No trial scores for %s/%s/%s/trial_%04d: %s",
            model_name, task_name, method, trial_idx, exc,
        )
        return None


def _load_baseline_pairs_from_hf(
    model_name: str, task_name: str,
) -> Optional[dict]:
    """Load baseline per-pair results from HF."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
        model_to_safe_name,
    )
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _hf_hub_download,
    )
    safe = model_to_safe_name(model_name)
    hf_path = f"baselines/{safe}/{task_name}/pair_results.json"
    try:
        local = _hf_hub_download(hf_path)
        with open(local, "r") as f:
            return json.load(f)
    except Exception as exc:
        _log.info(
            "No baseline pairs for %s/%s: %s",
            model_name, task_name, exc,
        )
        return None


def _truncate(text: str, length: int = DISPLAY_TRUNCATION_MEDIUM) -> str:
    """Truncate text for display."""
    if not text:
        return ""
    return text[:length] + "..." if len(text) > length else text


def load_response_comparison(
    model_name: str, task_name: str, best_method_results: dict,
) -> Optional[list[dict]]:
    """Load and compare baseline vs winning trial responses.

    Returns list of dicts with: prompt, baseline_response, baseline_eval,
    steered_response, steered_eval, status.
    """
    if not best_method_results:
        return None

    winner = best_method_results.get("winner")
    if not winner:
        return None

    method_results = best_method_results.get("method_results", {})
    winner_data = method_results.get(winner, {})
    all_trials = winner_data.get("all_trials", [])
    best_score = winner_data.get("best_score")
    best_trial_idx = None
    for i, trial in enumerate(all_trials):
        if trial.get("score") == best_score:
            best_trial_idx = i
            break
    if best_trial_idx is None:
        return None

    trial_evals = _load_trial_scores_from_hf(
        model_name, task_name, winner, best_trial_idx,
    )
    baseline_pairs = _load_baseline_pairs_from_hf(model_name, task_name)
    if not trial_evals or not baseline_pairs:
        return None

    baseline_by_prompt = {}
    for _hash, pair_data in baseline_pairs.items():
        prompt = pair_data.get("prompt", "")
        baseline_by_prompt[prompt] = pair_data

    comparisons = []
    for trial_eval in trial_evals:
        prompt = trial_eval.get("prompt", "")
        eval_data = trial_eval.get("evaluation", {})
        steered_correct = eval_data.get("correct", False)
        steered_response = trial_eval.get("generated_response", "")

        baseline = baseline_by_prompt.get(prompt, {})
        baseline_correct = baseline.get("correct", False)
        baseline_response = baseline.get("response", "")

        if steered_correct and not baseline_correct:
            status = "IMPROVED"
        elif not steered_correct and baseline_correct:
            status = "WORSENED"
        else:
            status = "unchanged"

        comparisons.append({
            "prompt": _truncate(prompt),
            "baseline_response": _truncate(baseline_response),
            "baseline_eval": "correct" if baseline_correct else "wrong",
            "steered_response": _truncate(steered_response),
            "steered_eval": "correct" if steered_correct else "wrong",
            "status": status,
        })

    return comparisons


def format_response_dataframe(
    comparisons: list[dict],
) -> list[list]:
    """Format comparisons as rows for gr.Dataframe."""
    rows = []
    for c in comparisons:
        rows.append([
            c["prompt"],
            c["baseline_eval"],
            c["steered_eval"],
            c["status"],
            c["baseline_response"],
            c["steered_response"],
        ])
    return rows


RESPONSE_COLUMNS = [
    "Prompt", "Baseline", "Steered", "Status",
    "Baseline Response", "Steered Response",
]
