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


def baseline_responses_hf_path(model: str, benchmark: str) -> str:
    """Build HF repo path for baseline (unsteered) responses JSON."""
    safe = model_to_safe_name(model)
    return f"baselines/{safe}/{benchmark}/responses.json"


def baseline_scores_hf_path(model: str, benchmark: str) -> str:
    """Build HF repo path for baseline (unsteered) scores JSON."""
    safe = model_to_safe_name(model)
    return f"baselines/{safe}/{benchmark}/scores.json"


def baseline_metadata_hf_path(model: str, benchmark: str) -> str:
    """Build HF repo path for baseline metadata JSON."""
    safe = model_to_safe_name(model)
    return f"baselines/{safe}/{benchmark}/metadata.json"


def personalization_baseline_hf_path(model: str) -> str:
    """Build HF repo path for personalization baseline responses JSON."""
    safe = model_to_safe_name(model)
    return f"baselines/{safe}/personalization/responses.json"
