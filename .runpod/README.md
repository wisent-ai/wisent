# Wisent Guard - Runpod Template

Latent space monitoring and guardrails for AI models. Detect hallucinations, harmful content, and steer model behavior using representation engineering.

## Features

- **Hallucination Detection**: Train classifiers on model activations to detect when LLMs are hallucinating
- **Representation Engineering**: Use contrastive pairs to identify and modify model behavior
- **Steering Vectors**: Create and apply steering vectors to guide model outputs
- **Interactive Notebooks**: Pre-installed Jupyter notebooks for learning and experimentation

## Quick Start

### Pod Deployment

1. Deploy this template as a Runpod Pod
2. Access JupyterLab at `http://<pod-ip>:8888`
3. Open `hallucination_guard.ipynb` to get started

### Serverless Deployment

Deploy as a serverless endpoint and use the API:

```python
import requests

# Health check
response = requests.post(
    "https://api.runpod.ai/v2/<endpoint_id>/runsync",
    headers={"Authorization": "Bearer <api_key>"},
    json={"input": {"action": "health"}}
)

# Train a hallucination classifier
response = requests.post(
    "https://api.runpod.ai/v2/<endpoint_id>/runsync",
    headers={"Authorization": "Bearer <api_key>"},
    json={
        "input": {
            "action": "train",
            "model_name": "Qwen/Qwen3-8B",
            "task": "truthfulqa_gen",
            "layer": 19,
            "training_limit": 100
        }
    }
)

# Classify text for hallucinations
response = requests.post(
    "https://api.runpod.ai/v2/<endpoint_id>/runsync",
    headers={"Authorization": "Bearer <api_key>"},
    json={
        "input": {
            "action": "classify",
            "text": "The capital of France is Berlin.",
            "question": "What is the capital of France?"
        }
    }
)
```

## API Reference

### Actions

| Action | Description |
|--------|-------------|
| `health` | Check service health and configuration |
| `generate_pairs` | Generate contrastive pairs from a benchmark |
| `train` | Train a hallucination classifier |
| `classify` | Classify text for hallucination probability |
| `create_steering_vector` | Create a steering vector |

### Parameters

#### `generate_pairs`
- `task` (string): Benchmark task name (default: "truthfulqa_gen")
- `limit` (int): Number of pairs to generate (default: 100)

#### `train`
- `model_name` (string): HuggingFace model ID
- `task` (string): Benchmark task name
- `layer` (int): Model layer for activation extraction
- `training_limit` (int): Number of training samples
- `classifier_type` (string): "logistic" or "mlp"

#### `classify`
- `text` (string): Text to classify
- `question` (string, optional): Original question for context
- `threshold` (float): Hallucination threshold (default: 0.5)

## Pre-installed Notebooks

| Notebook | Description |
|----------|-------------|
| `hallucination_guard.ipynb` | Complete tutorial on hallucination detection |
| `basics_of_representation_engineering.ipynb` | Fundamentals of representation engineering |
| `coding_boost.ipynb` | Steering for improved coding assistance |
| `abliteration.ipynb` | Feature removal and ablation techniques |
| `personalization_synthetic.ipynb` | Creating personalized models |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model ID | Qwen/Qwen3-8B |
| `TASK` | Benchmark task | truthfulqa_gen |
| `LAYER` | Model layer for activations | 19 |
| `TRAINING_LIMIT` | Training samples | 100 |
| `HF_TOKEN` | HuggingFace API token | - |
| `JUPYTER_PORT` | JupyterLab port | 8888 |
| `JUPYTER_PASSWORD` | JupyterLab password | - |

## Hardware Requirements

- **GPU**: RTX 4090, L40S, A100, or H100 recommended
- **VRAM**: 24GB+ for 8B models, 48GB+ for larger models
- **Disk**: 50GB+ for model caching

## Links

- [Wisent Documentation](https://wisent.ai/documentation)
- [GitHub Repository](https://github.com/Wisent-AI/wisent-guard)
- [PyPI Package](https://pypi.org/project/wisent/)

## Support

For issues and feature requests, please open an issue on the [GitHub repository](https://github.com/Wisent-AI/wisent-guard/issues).
