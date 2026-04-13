"""HuggingFace Hub configuration for activation storage."""

HF_REPO_ID = "wisent-ai/activations"
HF_REPO_TYPE = "dataset"


def model_to_safe_name(model_id: str) -> str:
    """Convert HuggingFace model ID to filesystem-safe name.

    Example: 'meta-llama/Llama-3.2-1B-Instruct' -> 'meta-llama__Llama-3.2-1B-Instruct'
    """
    return model_id.replace("/", "__")


def safe_name_to_model(safe_name: str) -> str:
    """Convert filesystem-safe name back to HuggingFace model ID.

    Example: 'meta-llama__Llama-3.2-1B-Instruct' -> 'meta-llama/Llama-3.2-1B-Instruct'
    """
    return safe_name.replace("__", "/")


def activation_hf_path(
    model: str, benchmark: str, strategy: str, layer: int
) -> str:
    """Build HF repo path for an activation shard."""
    safe = model_to_safe_name(model)
    return f"activations/{safe}/{benchmark}/{strategy}/layer_{layer}.safetensors"


def raw_activation_hf_path(
    model: str, benchmark: str, prompt_format: str, layer: int, chunk: int
) -> str:
    """Build HF repo path for a raw activation shard."""
    safe = model_to_safe_name(model)
    return f"raw_activations/{safe}/{benchmark}/{prompt_format}/layer_{layer}_chunk_{chunk}.safetensors"


def pair_texts_hf_path(benchmark: str) -> str:
    """Build HF repo path for pair texts JSON."""
    return f"pair_texts/{benchmark}.json"


def baseline_responses_hf_path(model: str, benchmark: str, pairs_hash: str = "") -> str:
    """Build HF repo path for baseline (unsteered) responses JSON."""
    safe = model_to_safe_name(model)
    sub = f"/{pairs_hash}" if pairs_hash else ""
    return f"baselines/{safe}/{benchmark}{sub}/responses.json"


def baseline_scores_hf_path(model: str, benchmark: str, pairs_hash: str = "") -> str:
    """Build HF repo path for baseline (unsteered) scores JSON."""
    safe = model_to_safe_name(model)
    sub = f"/{pairs_hash}" if pairs_hash else ""
    return f"baselines/{safe}/{benchmark}{sub}/scores.json"


def baseline_metadata_hf_path(model: str, benchmark: str, pairs_hash: str = "") -> str:
    """Build HF repo path for baseline metadata JSON."""
    safe = model_to_safe_name(model)
    sub = f"/{pairs_hash}" if pairs_hash else ""
    return f"baselines/{safe}/{benchmark}{sub}/metadata.json"


def test_results_hf_path(benchmark: str) -> str:
    """Build HF repo path for benchmark test results JSON."""
    return f"test_results/{benchmark}.json"


def viz_cache_hf_path(model: str, benchmark: str, layer: int) -> str:
    """Build HF repo path for visualization cache JSON."""
    safe = model_to_safe_name(model)
    return f"viz_cache/{safe}/{benchmark}/layer_{layer}.json"


def best_method_hf_path(model: str, benchmark: str) -> str:
    """Build HF repo path for find-best-method results JSON."""
    safe = model_to_safe_name(model)
    return f"best_method/{safe}/{benchmark}/results.json"


def personalization_baseline_hf_path(model: str) -> str:
    """Build HF repo path for personalization baseline responses JSON."""
    safe = model_to_safe_name(model)
    return f"baselines/{safe}/personalization/responses.json"
