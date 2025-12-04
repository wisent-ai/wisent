# Wisent - Runpod Pod Template

Run wisent on Runpod with JupyterLab and pre-installed example notebooks.

## Quick Start

### 1. Build the Docker image

```bash
cd wisent
docker build -t yourusername/wisent:latest -f .runpod/Dockerfile .
```

### 2. Push to Docker Hub

```bash
docker login
docker push yourusername/wisent:latest
```

### 3. Create Pod Template in Runpod

1. Go to [My Templates](https://console.runpod.io/user/templates)
2. Click **New Template**
3. Configure:
   - **Name**: `Wisent`
   - **Container Image**: `yourusername/wisent:latest`
   - **Container Disk**: 50 GB
   - **Volume Disk**: 100 GB (for model caching)
   - **Volume Mount Path**: `/workspace`
   - **HTTP Ports**: `8888` (JupyterLab)
   - **TCP Ports**: `22` (SSH)
4. Click **Save Template**

### 4. Deploy a Pod

1. Go to [Pods](https://console.runpod.io/pods)
2. Click **Deploy**
3. Select a GPU (RTX 4090, L40S, A100 recommended)
4. Choose your **Wisent** template
5. Click **Deploy On-Demand**

### 5. Connect

- **JupyterLab**: Click the JupyterLab link when ready
- **SSH**: Use the SSH command from the Pod details
- **Web Terminal**: Enable and open from Pod details

## Pre-installed Notebooks

Located in `/workspace/notebooks/`:

| Notebook | Description |
|----------|-------------|
| `hallucination_guard.ipynb` | Detect and prevent hallucinations |
| `basics_of_representation_engineering.ipynb` | Learn representation engineering fundamentals |
| `coding_boost.ipynb` | Steer models for better coding |
| `abliteration.ipynb` | Feature removal techniques |
| `personalization_synthetic.ipynb` | Create personalized models |

## Environment Variables (Optional)

Set these when creating your Pod template:

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token for gated models | `hf_xxx...` |
| `MODEL_NAME` | Default model to use | `Qwen/Qwen3-8B` |

## Hardware Recommendations

| Model Size | Recommended GPU | VRAM |
|------------|-----------------|------|
| 1-3B | RTX 4090 | 24 GB |
| 7-8B | L40S, A100 | 48 GB |
| 13B+ | A100 80GB, H100 | 80 GB |

## Links

- [Wisent Documentation](https://wisent.ai/documentation)
- [GitHub Repository](https://github.com/Wisent-AI/wisent)
