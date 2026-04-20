"""Per-pair baseline cache on HuggingFace.

Baselines are stored per-pair keyed by prompt hash at
baselines/{safe_model}/{benchmark}/pair_results.json.
Only uncached pairs are evaluated; cached results are reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    EVAL_F1_THRESHOLD,
    EVAL_GENERATION_EMBEDDING_WEIGHT,
    EVAL_GENERATION_NLI_WEIGHT,
    HASH_PREFIX_LEN,
    HF_RETRY_BACKOFF_MAX_EXPONENT,
    HF_RETRY_BASE_WAIT,
    HF_RETRY_JITTER_MAX,
    HF_RETRY_JITTER_MIN,
    HF_RETRY_MAX_RETRIES,
    HF_RETRY_RETRYABLE_PATTERNS,
    JSON_INDENT,
    RECURSION_INITIAL_DEPTH,
)
from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
    HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name,
)

logger = logging.getLogger(__name__)


def _pair_hash(prompt: str, expected: str) -> str:
    """Hash a pair by its prompt + expected answer."""
    key = f"{prompt}||{expected}"
    return hashlib.sha256(key.encode()).hexdigest()[:HASH_PREFIX_LEN]


def _pair_results_hf_path(model: str, benchmark: str) -> str:
    safe = model_to_safe_name(model)
    return f"baselines/{safe}/{benchmark}/pair_results.json"


def _load_cached_pair_results(model: str, benchmark: str) -> Dict[str, Dict]:
    """Download per-pair results from HF. Returns {pair_hash: result_dict}."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _get_hf_token,
    )
    from huggingface_hub import hf_hub_download
    hf_path = _pair_results_hf_path(model, benchmark)
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hf_path,
            repo_type=HF_REPO_TYPE, token=_get_hf_token(),
        )
        with open(local, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _upload_pair_results(
    model: str, benchmark: str, results: Dict[str, Dict],
) -> None:
    """Upload per-pair results to HF."""
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            _get_api, _retry_upload,
        )
        hf_path = _pair_results_hf_path(model, benchmark)
        retry_cfg = build_default_hf_retry_config()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as tmp:
            json.dump(results, tmp, indent=JSON_INDENT)
            tmp_path = tmp.name
        try:
            _retry_upload(
                lambda: _get_api().upload_file(
                    path_or_fileobj=tmp_path, path_in_repo=hf_path,
                    repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                ), **retry_cfg,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to upload pair results: %s", exc)


def build_default_hf_retry_config() -> Dict[str, Any]:
    """Build HF retry config dict from infrastructure constants."""
    return {
        "max_retries": HF_RETRY_MAX_RETRIES,
        "base_wait": HF_RETRY_BASE_WAIT,
        "backoff_max_exponent": HF_RETRY_BACKOFF_MAX_EXPONENT,
        "jitter_min": HF_RETRY_JITTER_MIN,
        "jitter_max": HF_RETRY_JITTER_MAX,
        "retryable_patterns": HF_RETRY_RETRYABLE_PATTERNS,
    }


def generate_baseline_with_cache(
    model: str, benchmark: str, pairs_file: str,
    device: Optional[str], cached_model=None,
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate unsteered model, reusing per-pair cache from HF.

    Returns (accuracy, responses_list, scores_list).
    """
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.models import get_generate_kwargs
    from wisent.core.reading.evaluators.rotator import EvaluatorRotator

    EvaluatorRotator.discover_evaluators("wisent.core.reading.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.reading.evaluators.benchmark_specific")

    with open(pairs_file) as f:
        pairs_data = json.load(f).get("pairs", [])
    if not pairs_data:
        raise ValueError(f"No pairs found in {pairs_file}")

    cached_results = _load_cached_pair_results(model, benchmark)
    cached_response_count = RECURSION_INITIAL_DEPTH

    # Identify which pairs need model generation (not evaluation — always re-evaluate)
    pair_hashes = []
    missing_indices = []
    for i, p in enumerate(pairs_data):
        pos = p.get("positive_response", {})
        expected = pos.get("model_response", "") if isinstance(pos, dict) else str(pos)
        ph = _pair_hash(p.get("prompt", ""), expected)
        pair_hashes.append(ph)
        if ph in cached_results and cached_results[ph].get("response"):
            cached_response_count += COMBO_OFFSET
        else:
            missing_indices.append(i)

    print(f"   Baseline: {cached_response_count} responses cached, "
          f"{len(missing_indices)} to generate", flush=True)

    # Ensure model available (needed for generation AND for log_likelihoods evaluation)
    wisent_model = cached_model if cached_model is not None else WisentModel(model, device=device)
    if missing_indices:
        gen_kwargs = get_generate_kwargs()
        for idx in missing_indices:
            pair_dict = pairs_data[idx]
            prompt = pair_dict.get("prompt", "")
            pos = pair_dict.get("positive_response", {})
            neg = pair_dict.get("negative_response", {})
            expected = pos.get("model_response", "") if isinstance(pos, dict) else str(pos)
            neg_resp = neg.get("model_response", "") if isinstance(neg, dict) else str(neg)
            response = wisent_model.generate(
                [[{"role": "user", "content": prompt}]], **gen_kwargs,
            )[RECURSION_INITIAL_DEPTH]
            cached_results[pair_hashes[idx]] = {
                "prompt": prompt, "response": response, "expected": expected,
                "negative": neg_resp,
            }
        _upload_pair_results(model, benchmark, cached_results)

    # Always re-evaluate all pairs with current evaluator
    evaluator = EvaluatorRotator(
        evaluator=None, task_name=benchmark, autoload=False,
        evaluator_kwargs={
            "f1_threshold": EVAL_F1_THRESHOLD,
            "generation_embedding_weight": EVAL_GENERATION_EMBEDDING_WEIGHT,
            "generation_nli_weight": EVAL_GENERATION_NLI_WEIGHT,
        },
    )
    responses_list, scores_list = [], []
    correct = RECURSION_INITIAL_DEPTH
    total = RECURSION_INITIAL_DEPTH
    for i, p in enumerate(pairs_data):
        r = cached_results.get(pair_hashes[i], {})
        response = r.get("response", "")
        expected = r.get("expected", "")
        neg_resp = r.get("negative", "")
        prompt = r.get("prompt", "")
        eval_kwargs = {
            "question": prompt, "task_name": benchmark,
            "correct_answers": [expected], "incorrect_answers": [neg_resp],
            "choices": [expected, neg_resp],
            "model": wisent_model,
        }
        result = evaluator.evaluate(response, expected, **eval_kwargs)
        is_correct = result.ground_truth == "TRUTHFUL"
        if is_correct:
            correct += COMBO_OFFSET
        total += COMBO_OFFSET
        responses_list.append({
            "prompt": prompt, "response": response,
            "expected": expected, "negative": neg_resp,
        })
        scores_list.append({
            "prompt": prompt, "correct": is_correct,
            "ground_truth": result.ground_truth,
        })
    if cached_model is None and wisent_model is not None:
        wisent_model.detach()
    accuracy = correct / total if total > RECURSION_INITIAL_DEPTH else RECURSION_INITIAL_DEPTH
    return accuracy, responses_list, scores_list
