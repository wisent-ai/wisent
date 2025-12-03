# Wisent Guard - Runpod Pod Template
# Latent space monitoring and guardrails for AI models

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/models

WORKDIR /app

# Copy wisent source code
COPY . /wisent

# Install all dependencies from requirements.txt first
RUN pip install -r /wisent/requirements.txt

# Install wisent in editable mode
RUN cd /wisent && pip install -e ".[harness]"

# Copy startup script (Option 2: run after base services start)
COPY .runpod/run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Use base image services (Jupyter/SSH) + our startup script
CMD ["/app/run.sh"]
