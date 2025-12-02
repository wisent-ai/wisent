"""
Wisent Guard - Runpod Serverless Handler

This handler provides a serverless API for wisent-guard functionality:
- Health checks
- Generating contrastive pairs from benchmarks
- Training hallucination classifiers
- Classifying text for hallucinations
- Creating steering vectors
"""

import os
import json
import torch
import runpod
from typing import Any

# Global model and classifier cache
_model_cache = {}
_classifier_cache = {}


def get_model(model_name: str):
    """Load and cache a model."""
    if model_name not in _model_cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        _model_cache[model_name] = {"model": model, "tokenizer": tokenizer}
        print(f"Model loaded: {model_name}")

    return _model_cache[model_name]


def health_check() -> dict:
    """Perform a health check."""
    import wisent

    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "status": "healthy",
        "wisent_version": wisent.__version__,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "model_name": os.environ.get("MODEL_NAME", "not set"),
    }


def generate_pairs(task: str, limit: int = 100, output_path: str = None) -> dict:
    """Generate contrastive pairs from a benchmark task."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
        build_contrastive_pairs_from_task
    )

    print(f"Generating {limit} contrastive pairs from task: {task}")

    pairs = build_contrastive_pairs_from_task(
        task_name=task,
        limit=limit
    )

    result = {
        "task": task,
        "num_pairs": len(pairs),
        "pairs": [p.to_dict() if hasattr(p, 'to_dict') else vars(p) for p in pairs[:10]],  # Return first 10 as sample
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump({"pairs": [p.to_dict() if hasattr(p, 'to_dict') else vars(p) for p in pairs]}, f)
        result["output_path"] = output_path

    return result


def train_classifier(
    model_name: str = None,
    task: str = None,
    layer: int = None,
    training_limit: int = 100,
    classifier_type: str = "logistic"
) -> dict:
    """Train a hallucination classifier."""
    import subprocess
    import sys

    model_name = model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    task = task or os.environ.get("TASK", "truthfulqa_gen")
    layer = layer or int(os.environ.get("LAYER", "19"))

    output_dir = "/workspace/outputs/classifier"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training classifier for {model_name} on {task}, layer {layer}")

    cmd = [
        sys.executable, "-m", "wisent.core.main", "tasks", task,
        "--model", model_name,
        "--layer", str(layer),
        "--classifier-type", classifier_type,
        "--limit", str(training_limit),
        "--save-classifier", f"{output_dir}/classifier.pt",
        "--output", output_dir,
        "--verbose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")

    if result.returncode != 0:
        return {
            "status": "error",
            "error": result.stderr,
            "stdout": result.stdout
        }

    return {
        "status": "success",
        "model": model_name,
        "task": task,
        "layer": layer,
        "classifier_type": classifier_type,
        "output_dir": output_dir,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
    }


def classify_text(
    text: str,
    question: str = None,
    model_name: str = None,
    layer: int = None,
    threshold: float = 0.5
) -> dict:
    """Classify text for hallucination probability."""
    model_name = model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    layer = layer or int(os.environ.get("LAYER", "19"))

    # Check if we have a trained classifier
    classifier_path = "/workspace/outputs/classifier/classifier.pt"

    if not os.path.exists(classifier_path):
        return {
            "status": "error",
            "error": "No classifier found. Please train a classifier first using action='train'."
        }

    # Load model and classifier
    model_data = get_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # Prepare input
    if question:
        prompt = f"Question: {question}\nAnswer: {text}"
    else:
        prompt = text

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get activations at the specified layer
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        activation = hidden_states[:, -1, :].cpu().numpy()  # Last token

    # Load and apply classifier
    classifier = torch.load(classifier_path, weights_only=False)

    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(activation)[0]
        hallucination_prob = proba[0] if len(proba) > 1 else 1 - proba[0]
    else:
        prediction = classifier.predict(activation)[0]
        hallucination_prob = float(prediction)

    is_hallucination = hallucination_prob > threshold

    return {
        "status": "success",
        "text": text[:200] + "..." if len(text) > 200 else text,
        "hallucination_probability": float(hallucination_prob),
        "is_hallucination": is_hallucination,
        "threshold": threshold,
        "model": model_name,
        "layer": layer
    }


def create_steering_vector(
    model_name: str = None,
    task: str = None,
    layer: int = None,
    training_limit: int = 100
) -> dict:
    """Create a steering vector for the model."""
    import subprocess
    import sys

    model_name = model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    task = task or os.environ.get("TASK", "truthfulqa_gen")
    layer = layer or int(os.environ.get("LAYER", "19"))

    output_dir = "/workspace/outputs/steering"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating steering vector for {model_name} on {task}, layer {layer}")

    cmd = [
        sys.executable, "-m", "wisent.core.main", "generate-vector-from-task",
        task,
        "--model", model_name,
        "--layer", str(layer),
        "--limit", str(training_limit),
        "--output", f"{output_dir}/steering_vector.pt",
        "--verbose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")

    if result.returncode != 0:
        return {
            "status": "error",
            "error": result.stderr,
            "stdout": result.stdout
        }

    return {
        "status": "success",
        "model": model_name,
        "task": task,
        "layer": layer,
        "output_path": f"{output_dir}/steering_vector.pt",
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
    }


def handler(job: dict) -> Any:
    """
    Runpod serverless handler function.

    Supported actions:
    - health: Health check
    - generate_pairs: Generate contrastive pairs from a benchmark
    - train: Train a hallucination classifier
    - classify: Classify text for hallucinations
    - create_steering_vector: Create a steering vector
    """
    job_input = job.get("input", {})
    action = job_input.get("action", "health")

    try:
        if action == "health":
            return health_check()

        elif action == "generate_pairs":
            return generate_pairs(
                task=job_input.get("task", os.environ.get("TASK", "truthfulqa_gen")),
                limit=job_input.get("limit", 100),
                output_path=job_input.get("output_path")
            )

        elif action == "train":
            return train_classifier(
                model_name=job_input.get("model_name"),
                task=job_input.get("task"),
                layer=job_input.get("layer"),
                training_limit=job_input.get("training_limit", 100),
                classifier_type=job_input.get("classifier_type", "logistic")
            )

        elif action == "classify":
            return classify_text(
                text=job_input.get("text", ""),
                question=job_input.get("question"),
                model_name=job_input.get("model_name"),
                layer=job_input.get("layer"),
                threshold=job_input.get("threshold", 0.5)
            )

        elif action == "create_steering_vector":
            return create_steering_vector(
                model_name=job_input.get("model_name"),
                task=job_input.get("task"),
                layer=job_input.get("layer"),
                training_limit=job_input.get("training_limit", 100)
            )

        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "supported_actions": ["health", "generate_pairs", "train", "classify", "create_steering_vector"]
            }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start the Runpod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
