# Wisent Guard - Runpod Pod Template
#
# Build and push to Docker Hub:
#   docker build -t yourusername/wisent-guard:latest -f .runpod/Dockerfile .
#   docker push yourusername/wisent-guard:latest
#
# Then create a Pod template in Runpod console pointing to your image.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/models \
    TRANSFORMERS_CACHE=/workspace/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directories
RUN mkdir -p /workspace/notebooks /workspace/outputs /workspace/models

# Copy wisent-guard source and install
# Copy only necessary source files and setup
COPY wisent/ /tmp/wisent/wisent/
COPY setup.py pyproject.toml requirements.txt /tmp/wisent/
RUN cd /tmp/wisent && \
    pip install --no-cache-dir -e ".[harness]" && \
    pip install --no-cache-dir jupyterlab ipywidgets accelerate


# Copy example notebooks to workspace
RUN cp -r /tmp/wisent/wisent/examples/notebooks/* /workspace/notebooks/ 2>/dev/null || true

# Set working directory
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888

# Default command - the base image handles JupyterLab startup
