"""Response diff analysis for find_best_method.

Compares the winning method's best trial against unsteered baseline
to identify which questions flipped correct/wrong after steering.
"""
import json
import os
from typing import Any, Dict, List

from wisent.core.utils.config_tools.constants import (
    DISPLAY_TRUNCATION_MEDIUM,
    JSON_INDENT,
    RECURSION_INITIAL_DEPTH,
    SCORE_RANGE_MIN,
)
from wisent.core.utils.cli.optimize_steering.pipeline.comprehensive import (
    baseline_cache,
)


def build_winner_diff(
    output_dir: str,
    winner_method: str,
    model_name: str,
    benchmark: str,
) -> Dict[str, Any]:
    """Compare winner's best trial responses against baseline.

    Returns dict with flipped_correct, flipped_wrong, unchanged counts
    and per-question details.
    """
    best_trial_dir = _find_best_trial_dir(output_dir, winner_method)
    if not best_trial_dir:
        return {"error": "No trial directories found"}

    trial_scores = _load_trial_scores(best_trial_dir)
    if not trial_scores:
        return {"error": "Could not load trial scores"}

    baseline_scores = _load_baseline_scores(model_name, benchmark)
    if not baseline_scores:
        return {"error": "Could not load baseline scores"}

    baseline_map = {s["prompt"]: s["correct"] for s in baseline_scores}
    trial_map = {}
    for ev in trial_scores.get("evaluations", []):
        prompt = ev.get("prompt", "")
        correct = ev.get("evaluation", {}).get("correct", False)
        trial_map[prompt] = correct

    flipped_correct = RECURSION_INITIAL_DEPTH
    flipped_wrong = RECURSION_INITIAL_DEPTH
    unchanged = RECURSION_INITIAL_DEPTH
    details: List[Dict[str, Any]] = []

    for prompt in trial_map:
        baseline_correct = baseline_map.get(prompt)
        trial_correct = trial_map[prompt]

        if baseline_correct is None:
            continue

        if not baseline_correct and trial_correct:
            flipped_correct += True
            details.append({
                "prompt": prompt[:DISPLAY_TRUNCATION_MEDIUM],
                "flip": "wrong_to_correct",
            })
        elif baseline_correct and not trial_correct:
            flipped_wrong += True
            details.append({
                "prompt": prompt[:DISPLAY_TRUNCATION_MEDIUM],
                "flip": "correct_to_wrong",
            })
        else:
            unchanged += True

    result = {
        "flipped_correct": flipped_correct,
        "flipped_wrong": flipped_wrong,
        "unchanged": unchanged,
        "net_improvement": flipped_correct - flipped_wrong,
        "details": details,
    }

    diff_path = os.path.join(
        output_dir, f"winner_diff_{benchmark}.json",
    )
    with open(diff_path, "w") as f:
        json.dump(result, f, indent=JSON_INDENT)

    return result


def _find_best_trial_dir(output_dir: str, method: str) -> str:
    """Find the trial directory with the highest score for a method."""
    trials_dir = os.path.join(output_dir, "trials", method)
    if not os.path.isdir(trials_dir):
        return ""

    best_score = SCORE_RANGE_MIN
    best_dir = ""
    for entry in os.listdir(trials_dir):
        trial_path = os.path.join(trials_dir, entry)
        if not os.path.isdir(trial_path) or entry.startswith("_"):
            continue
        meta_path = os.path.join(trial_path, "trial_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        score = meta.get("score", SCORE_RANGE_MIN)
        if score > best_score:
            best_score = score
            best_dir = trial_path

    return best_dir


def _load_trial_scores(trial_dir: str) -> Dict[str, Any]:
    """Load scores.json from a trial directory."""
    scores_path = os.path.join(trial_dir, "scores.json")
    if not os.path.exists(scores_path):
        return {}
    with open(scores_path) as f:
        return json.load(f)


def _load_baseline_scores(
    model_name: str, benchmark: str,
) -> List[Dict[str, Any]]:
    """Load baseline scores from HF cache. Returns [] if cache helpers are unavailable."""
    check_fn = getattr(baseline_cache, "check_baseline_exists", None)
    load_fn = getattr(baseline_cache, "load_baseline_from_hf", None)
    if check_fn is None or load_fn is None:
        return []
    if not check_fn(model_name, benchmark):
        return []
    _, scores, _ = load_fn(model_name, benchmark)
    return scores
