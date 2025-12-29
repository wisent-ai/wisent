"""
Shared utilities for comparison experiments.
"""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from wisent.core.utils.device import preferred_dtype

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel


# SAE configurations for supported Gemma models
SAE_CONFIGS = {
    "google/gemma-2-2b": {
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "num_layers": 26,
        "default_layer": 12,
        "d_model": 2304,
        "d_sae": 16384,
    },
    "google/gemma-2-9b": {
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "num_layers": 42,
        "default_layer": 12,
        "d_model": 3584,
        "d_sae": 16384,
    },
}


def load_sae(model_name: str, layer_idx: int, device: str = "cuda:0"):
    """
    Load Gemma Scope SAE for a specific layer.

    Args:
        model_name: HuggingFace model name (e.g., 'google/gemma-2-2b')
        layer_idx: Layer index to load SAE for
        device: Device to load SAE on

    Returns:
        Tuple of (SAE object, sparsity tensor)
    """
    from sae_lens import SAE

    if model_name not in SAE_CONFIGS:
        raise ValueError(f"No SAE config for model '{model_name}'. Supported: {list(SAE_CONFIGS.keys())}")

    config = SAE_CONFIGS[model_name]
    sae_id = config["sae_id_template"].format(layer=layer_idx)

    print(f"   Loading SAE from {config['sae_release']} / {sae_id}")
    sae, _, sparsity = SAE.from_pretrained(
        release=config["sae_release"],
        sae_id=sae_id,
        device=device,
    )

    return sae, sparsity


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda:0",
    eval_mode: bool = True,
) -> tuple:
    """
    Load HuggingFace model and tokenizer.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        eval_mode: Whether to set model to eval mode (default True for inference)

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=preferred_dtype(device),
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if eval_mode:
        model.eval()

    return model, tokenizer


def generate_contrastive_pairs(
    task: str,
    num_pairs: int,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[list[dict], str]:
    """
    Generate contrastive pairs from an lm-eval task.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        num_pairs: Number of pairs to generate
        seed: Random seed for reproducibility
        verbose: Whether to print verbose output

    Returns:
        Tuple of (pairs list, path to temporary pairs file).
        Caller is responsible for cleaning up the file if needed.
    """
    from wisent.core.cli.generate_pairs_from_task import execute_generate_pairs_from_task

    pairs_file = tempfile.NamedTemporaryFile(mode='w', suffix='_pairs.json', delete=False).name
    pairs_args = Namespace(
        task_name=task,
        limit=num_pairs,
        output=pairs_file,
        seed=seed,
        verbose=verbose,
    )
    execute_generate_pairs_from_task(pairs_args)

    with open(pairs_file) as f:
        pairs_data = json.load(f)
    pairs = pairs_data["pairs"]

    return pairs, pairs_file


def create_test_only_task(task_name: str, train_ratio: float = 0.8) -> dict:
    """
    Create a task that evaluates only on our test split.

    This ensures no overlap with the data used for steering vector training.

    Args:
        task_name: lm-eval task name (e.g., 'boolq', 'cb')
        train_ratio: Fraction of data used for training (default 0.8)

    Returns:
        Task dict with test split configured
    """
    from lm_eval.tasks import get_task_dict
    from wisent.core.utils.dataset_splits import get_test_docs

    task_dict = get_task_dict([task_name])
    task = task_dict[task_name]

    test_docs = get_test_docs(task, benchmark_name=task_name, train_ratio=train_ratio)
    test_pct = round((1 - train_ratio) * 100)

    print(f"Test split size: {len(test_docs)} docs ({test_pct}% of pooled data)")

    # Override task's doc methods to use our test split
    task.test_docs = lambda: test_docs
    task.has_test_docs = lambda: True
    task._eval_docs = test_docs

    return {task_name: task}


def extract_accuracy(results: dict, task: str) -> float:
    """
    Extract accuracy from lm-eval results.

    Args:
        results: Results dict from lm-eval evaluator
        task: Task name to extract accuracy for

    Returns:
        Accuracy value (0.0 if not found)
    """
    task_results = results.get("results", {}).get(task, {})
    for key in ["acc", "acc,none", "accuracy", "acc_norm", "acc_norm,none"]:
        if key in task_results:
            return task_results[key]
    return 0.0


def run_lm_eval_evaluation(
    wisent_model: "WisentModel",
    task_dict: dict,
    task_name: str,
    batch_size: int | str = 1,
    max_batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    """
    Run evaluation using lm-eval-harness.

    Args:
        wisent_model: WisentModel instance
        task_dict: Task dict from create_test_only_task
        task_name: lm-eval task name
        batch_size: Batch size for evaluation
        max_batch_size: Max batch size for lm-eval internal batching
        limit: Max number of examples to evaluate

    Returns:
        Full results dict from lm-eval
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=wisent_model.hf_model,
        tokenizer=wisent_model.tokenizer,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
    )

    return results


def run_ll_evaluation(
    wisent_model: "WisentModel",
    task_dict: dict,
    task_name: str,
    limit: int | None = None,
) -> float:
    """
    Run evaluation using wisent's LogLikelihoodsEvaluator.

    Args:
        wisent_model: WisentModel instance
        task_dict: Task dict from create_test_only_task
        task_name: lm-eval task name
        limit: Max number of examples to evaluate

    Returns:
        Accuracy as float
    """
    from wisent.core.evaluators.benchmark_specific.log_likelihoods_evaluator import LogLikelihoodsEvaluator
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor

    ll_evaluator = LogLikelihoodsEvaluator()
    extractor = get_extractor(task_name)

    task = task_dict[task_name]
    docs = list(task.test_docs())

    if limit:
        docs = docs[:limit]

    print(f"Evaluating {len(docs)} examples with LogLikelihoodsEvaluator")

    correct = 0
    for i, doc in enumerate(docs):
        question = task.doc_to_text(doc)
        choices, expected = extractor.extract_choices_and_answer(task, doc)

        result = ll_evaluator.evaluate(
            response="",
            expected=expected,
            model=wisent_model,
            question=question,
            choices=choices,
            task_name=task_name,
        )

        if result.ground_truth == "TRUTHFUL":
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(docs)}, acc: {correct/(i+1):.4f}")

    return correct / len(docs) if docs else 0.0


def load_steering_vector(path: str | Path, default_method: str = "unknown") -> dict:
    """
    Load a steering vector from file.

    Args:
        path: Path to steering vector file (.json or .pt)
        default_method: Default method name if not found in file

    Returns:
        Dictionary with steering vectors and metadata
    """
    path = Path(path)

    if path.suffix == ".pt":
        from wisent.core.utils.device import resolve_default_device
        data = torch.load(path, map_location=resolve_default_device(), weights_only=False)
        layer_idx = str(data.get("layer_index", data.get("layer", 1)))
        return {
            "steering_vectors": {layer_idx: data["steering_vector"].tolist()},
            "layers": [layer_idx],
            "model": data.get("model", "unknown"),
            "method": data.get("method", default_method),
            "trait_label": data.get("trait_label", "unknown"),
        }
    else:
        with open(path) as f:
            return json.load(f)


def apply_steering_to_model(
    model: "WisentModel",
    steering_data: dict,
    scale: float = 1.0,
) -> None:
    """
    Apply loaded steering vectors to a WisentModel.

    Args:
        model: WisentModel instance
        steering_data: Dictionary from load_steering_vector()
        scale: Scaling factor for steering strength
    """
    raw_map = {}
    dtype = preferred_dtype()
    for layer_str, vec_list in steering_data["steering_vectors"].items():
        raw_map[layer_str] = torch.tensor(vec_list, dtype=dtype)

    model.set_steering_from_raw(raw_map, scale=scale, normalize=False)
    model.apply_steering()


def remove_steering(model: "WisentModel") -> None:
    """Remove steering from a WisentModel."""
    model.detach()
    model.clear_steering()


def convert_to_lm_eval_format(
    steering_data: dict,
    output_path: str | Path,
    scale: float = 1.0,
) -> Path:
    """
    Convert our steering vector format to lm-eval's steered model format.

    lm-eval expects:
    {
        "layers.N": {
            "steering_vector": tensor of shape (1, hidden_dim),
            "steering_coefficient": float,
            "action": "add"
        }
    }
    """
    output_path = Path(output_path)

    dtype = preferred_dtype()
    lm_eval_config = {}
    for layer_str, vec_list in steering_data["steering_vectors"].items():
        vec = torch.tensor(vec_list, dtype=dtype)
        # lm-eval expects shape (1, hidden_dim)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        layer_key = f"layers.{layer_str}"
        lm_eval_config[layer_key] = {
            "steering_vector": vec,
            "steering_coefficient": scale,
            "action": "add",
        }

    torch.save(lm_eval_config, output_path)
    return output_path
