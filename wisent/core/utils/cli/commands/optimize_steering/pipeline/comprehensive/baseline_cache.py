"""HuggingFace baseline response cache for steering optimization.

Check, load, and generate+upload unsteered model baselines.
Baselines are stored on the wisent-ai/activations HF dataset repo
under baselines/{safe_model}/{benchmark}/.
"""
from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    EVAL_F1_THRESHOLD,
    EVAL_GENERATION_EMBEDDING_WEIGHT,
    EVAL_GENERATION_NLI_WEIGHT,
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
    HF_REPO_ID,
    HF_REPO_TYPE,
    baseline_metadata_hf_path,
    baseline_responses_hf_path,
    baseline_scores_hf_path,
    model_to_safe_name,
    personalization_baseline_hf_path,
)

logger = logging.getLogger(__name__)


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


def check_baseline_exists(model: str, benchmark: str) -> bool:
    """Check if a cached baseline exists on HuggingFace for this model+benchmark."""
    from huggingface_hub import hf_hub_download
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _get_hf_token,
    )

    hf_path = baseline_responses_hf_path(model, benchmark)
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type=HF_REPO_TYPE,
            token=_get_hf_token(),
        )
        return True
    except Exception:
        return False


def load_baseline_from_hf(
    model: str, benchmark: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load cached baseline responses, scores, and metadata from HuggingFace.

    Returns:
        Tuple of (responses_list, scores_list, metadata_dict).
    """
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _hf_hub_download,
    )

    responses_path = _hf_hub_download(baseline_responses_hf_path(model, benchmark))
    with open(responses_path, "r") as f:
        responses_data = json.load(f)

    scores_path = _hf_hub_download(baseline_scores_hf_path(model, benchmark))
    with open(scores_path, "r") as f:
        scores_data = json.load(f)

    try:
        meta_path = _hf_hub_download(baseline_metadata_hf_path(model, benchmark))
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}

    return (
        responses_data.get("responses", []),
        scores_data.get("scores", []),
        metadata,
    )


def generate_and_upload_baseline(
    model: str,
    benchmark: str,
    pairs_file: str,
    device: Optional[str],
    hf_retry_config: Dict[str, Any],
    cached_model=None,
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate baseline responses, evaluate, upload to HF, return results.

    Returns:
        Tuple of (accuracy_score, responses_list, scores_list).
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
        raise ValueError(f"No pairs found in {pairs_file}")

    wisent_model = cached_model if cached_model is not None else WisentModel(model, device=device)
    evaluator = EvaluatorRotator(
        evaluator=None, task_name=benchmark, autoload=False,
        evaluator_kwargs={
            "f1_threshold": EVAL_F1_THRESHOLD,
            "generation_embedding_weight": EVAL_GENERATION_EMBEDDING_WEIGHT,
            "generation_nli_weight": EVAL_GENERATION_NLI_WEIGHT,
        },
    )
    gen_kwargs = get_generate_kwargs()

    responses_list: List[Dict[str, Any]] = []
    scores_list: List[Dict[str, Any]] = []
    correct = RECURSION_INITIAL_DEPTH
    total = RECURSION_INITIAL_DEPTH

    for pair_dict in pairs_data:
        prompt = pair_dict.get("prompt", "")
        positive = pair_dict.get("positive_response", {})
        negative = pair_dict.get("negative_response", {})
        expected = positive.get("model_response", "")
        neg_response = negative.get("model_response", "")

        messages = [{"role": "user", "content": prompt}]
        response = wisent_model.generate(
            [messages], **gen_kwargs,
        )[RECURSION_INITIAL_DEPTH]

        eval_kwargs = {
            "response": response,
            "expected": expected,
            "question": prompt,
            "choices": [neg_response, expected],
            "task_name": benchmark,
            "model": wisent_model,
        }
        metadata = pair_dict.get("metadata", {})
        if metadata:
            for key, value in metadata.items():
                if value is not None and key not in eval_kwargs:
                    eval_kwargs[key] = value

        result = evaluator.evaluate(**eval_kwargs)
        is_correct = result.ground_truth == "TRUTHFUL"
        if is_correct:
            correct += COMBO_OFFSET
        total += COMBO_OFFSET

        responses_list.append({
            "prompt": prompt,
            "response": response,
            "expected": expected,
            "negative": neg_response,
        })
        scores_list.append({
            "prompt": prompt,
            "correct": is_correct,
            "ground_truth": result.ground_truth,
        })

    if cached_model is None:
        wisent_model.detach()

    accuracy = correct / total if total > RECURSION_INITIAL_DEPTH else RECURSION_INITIAL_DEPTH

    _upload_baseline(
        model, benchmark, responses_list, scores_list,
        accuracy, total, hf_retry_config,
    )

    return accuracy, responses_list, scores_list


def _upload_baseline(
    model: str,
    benchmark: str,
    responses_list: List[Dict[str, Any]],
    scores_list: List[Dict[str, Any]],
    accuracy: float,
    total: int,
    hf_retry_config: Dict[str, Any],
) -> None:
    """Upload baseline responses, scores, and metadata to HuggingFace."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
        _get_api,
        _retry_upload,
    )

    safe = model_to_safe_name(model)

    files_to_upload = [
        (
            baseline_responses_hf_path(model, benchmark),
            {"model": model, "benchmark": benchmark, "responses": responses_list},
        ),
        (
            baseline_scores_hf_path(model, benchmark),
            {"model": model, "benchmark": benchmark, "scores": scores_list},
        ),
        (
            baseline_metadata_hf_path(model, benchmark),
            {
                "model": model,
                "safe_model": safe,
                "benchmark": benchmark,
                "accuracy": accuracy,
                "total_pairs": total,
                "timestamp": datetime.now().isoformat(),
            },
        ),
    ]

    api = _get_api()
    for hf_path, payload in files_to_upload:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=JSON_INDENT)
            tmp_path = tmp.name
        try:
            _retry_upload(
                lambda p=tmp_path, hp=hf_path: api.upload_file(
                    path_or_fileobj=p,
                    path_in_repo=hp,
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                ),
                **hf_retry_config,
            )
            logger.info("Uploaded baseline: %s", hf_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
