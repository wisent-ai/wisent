# Wisent Guard - Runpod Template
# Latent space monitoring and guardrails for AI models
#
# This Dockerfile sets up:
# - CUDA 12.1 with PyTorch 2.5.1
# - wisent-guard with all dependencies
# - JupyterLab for interactive notebooks
# - Pre-installed example notebooks

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create working directories
WORKDIR /app
RUN mkdir -p /workspace/notebooks /workspace/outputs /workspace/models

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy wisent-guard source code
COPY . /wisent-guard

# Install wisent-guard with all dependencies
RUN cd /wisent-guard && pip install -e ".[harness]"

# Install additional dependencies for notebooks and Runpod
RUN pip install \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    runpod>=1.6.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    flash-attn>=2.5.0 --no-build-isolation || true

# Fix datasets version for compatibility
RUN pip install --force-reinstall datasets==3.6.0

# Copy example notebooks to workspace
RUN cp -r /wisent-guard/wisent/examples/notebooks/* /workspace/notebooks/ 2>/dev/null || true

# Copy Runpod handler
COPY .runpod/handler.py /app/handler.py
COPY .runpod/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set environment variables
ENV HF_HOME=/workspace/models \
    TRANSFORMERS_CACHE=/workspace/models \
    JUPYTER_PORT=8888 \
    MODEL_NAME=Qwen/Qwen3-8B \
    TASK=truthfulqa_gen \
    LAYER=19 \
    TRAINING_LIMIT=100

# Expose ports for Jupyter and API
EXPOSE 8888 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both Jupyter and the handler
CMD ["/app/start.sh"]
