"""Response evaluation for steering optimization - wraps wisent evaluate-responses CLI."""
import importlib as _importlib
import json as _json

_mod = _importlib.import_module(
    ".analysis.evaluation.evaluate_responses", "wisent.core.utils.cli",
)
execute_evaluate_responses = _mod.execute_evaluate_responses


def task_uses_log_likelihoods(task_name: str) -> bool:
    """Return True if the task's extractor declares evaluator_name='log_likelihoods'."""
    try:
        from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor as _g
        e = _g(task_name)
    except Exception:
        try:
            from wisent.extractors.hf.hf_extractor_registry import get_extractor as _g
            e = _g(task_name)
        except Exception:
            return False
    return getattr(e, "evaluator_name", "") == "log_likelihoods"


def write_placeholder_responses(pairs_file: str, responses_file: str, limit: int, task: str, model: str) -> None:
    """Write a responses.json with empty generated text so log_likelihoods can evaluate without generation."""
    with open(pairs_file) as f:
        pairs = _json.load(f).get("pairs", [])[:limit]
    responses = []
    for p in pairs:
        pos = p.get("positive_response", {}).get("model_response", "") if isinstance(p.get("positive_response"), dict) else str(p.get("positive_response", ""))
        neg = p.get("negative_response", {}).get("model_response", "") if isinstance(p.get("negative_response"), dict) else str(p.get("negative_response", ""))
        responses.append({
            "prompt": p.get("prompt", ""), "generated_response": "",
            "positive_reference": pos, "negative_reference": neg,
            "correct_answers": [pos], "incorrect_answers": [neg],
        })
    with open(responses_file, "w") as f:
        _json.dump({"task": task, "model": model, "responses": responses}, f)


__all__ = ["execute_evaluate_responses", "task_uses_log_likelihoods", "write_placeholder_responses"]
