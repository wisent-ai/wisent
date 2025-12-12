# Dockerfile for running lm-evaluation-harness HumanEval safely
# Build: docker build -t lm-eval:humaneval .
# Run: see run_humaneval.sh

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create unprivileged user
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /home/sandbox

# Create virtual environment
RUN python -m venv /home/sandbox/venv
ENV PATH="/home/sandbox/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install lm-evaluation-harness and evalplus for humaneval_plus
RUN pip install --no-cache-dir \
    lm-eval \
    transformers \
    accelerate \
    datasets \
    sentencepiece \
    evalplus

# Environment for code evaluation
ENV HF_ALLOW_CODE_EVAL=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/sandbox/.cache/huggingface

RUN mkdir -p /home/sandbox/.cache/huggingface /home/sandbox/output

WORKDIR /home/sandbox
ENTRYPOINT ["lm_eval"]
