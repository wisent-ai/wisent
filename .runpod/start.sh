#!/bin/bash
# Wisent Guard - Runpod Startup Script
# This script starts both JupyterLab and the serverless handler

set -e

echo "=============================================="
echo "  Wisent Guard - Runpod Template"
echo "=============================================="
echo ""

# Display configuration
echo "Configuration:"
echo "  MODEL_NAME: ${MODEL_NAME:-Qwen/Qwen3-8B}"
echo "  TASK: ${TASK:-truthfulqa_gen}"
echo "  LAYER: ${LAYER:-19}"
echo "  JUPYTER_PORT: ${JUPYTER_PORT:-8888}"
echo ""

# Set HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
    echo "Setting HuggingFace token..."
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
fi

# Create output directories
mkdir -p /workspace/notebooks /workspace/outputs /workspace/models

# Copy notebooks if they don't exist
if [ ! -f "/workspace/notebooks/hallucination_guard.ipynb" ]; then
    echo "Copying example notebooks to /workspace/notebooks..."
    cp -r /wisent-guard/wisent/examples/notebooks/* /workspace/notebooks/ 2>/dev/null || true
fi

# Set up Jupyter password if provided
JUPYTER_ARGS="--ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} --no-browser --allow-root"
JUPYTER_ARGS="$JUPYTER_ARGS --notebook-dir=/workspace/notebooks"
JUPYTER_ARGS="$JUPYTER_ARGS --ServerApp.allow_origin='*'"
JUPYTER_ARGS="$JUPYTER_ARGS --ServerApp.allow_remote_access=True"

if [ -n "$JUPYTER_PASSWORD" ]; then
    echo "Setting Jupyter password..."
    JUPYTER_ARGS="$JUPYTER_ARGS --ServerApp.password=$(python -c "from jupyter_server.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))")"
else
    JUPYTER_ARGS="$JUPYTER_ARGS --ServerApp.token=''"
fi

# Function to start JupyterLab
start_jupyter() {
    echo ""
    echo "Starting JupyterLab on port ${JUPYTER_PORT:-8888}..."
    jupyter lab $JUPYTER_ARGS &
    JUPYTER_PID=$!
    echo "JupyterLab started with PID: $JUPYTER_PID"
}

# Function to start the serverless handler
start_handler() {
    echo ""
    echo "Starting Runpod serverless handler..."
    cd /app
    python handler.py &
    HANDLER_PID=$!
    echo "Handler started with PID: $HANDLER_PID"
}

# Detect if running in serverless mode or Pod mode
if [ "$RUNPOD_SERVERLESS" = "true" ] || [ -n "$RUNPOD_POD_ID" ]; then
    echo ""
    echo "Detected Runpod environment..."

    # In serverless mode, only run the handler
    if [ "$RUNPOD_SERVERLESS" = "true" ]; then
        echo "Running in serverless mode - starting handler only"
        cd /app
        exec python handler.py
    else
        # In Pod mode, run both Jupyter and handler
        echo "Running in Pod mode - starting Jupyter and handler"
        start_jupyter
        start_handler
    fi
else
    # Local/development mode - run both
    echo "Running in local mode - starting Jupyter and handler"
    start_jupyter
    start_handler
fi

echo ""
echo "=============================================="
echo "  Wisent Guard is ready!"
echo "=============================================="
echo ""
echo "Access points:"
echo "  JupyterLab: http://localhost:${JUPYTER_PORT:-8888}"
echo "  API: http://localhost:8000"
echo ""
echo "Example notebooks available in /workspace/notebooks:"
echo "  - hallucination_guard.ipynb"
echo "  - basics_of_representation_engineering.ipynb"
echo "  - coding_boost.ipynb"
echo "  - abliteration.ipynb"
echo "  - personalization_synthetic.ipynb"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for any process to exit
wait -n

# If any process exits, kill all and exit
echo "A service exited, shutting down..."
kill $JUPYTER_PID $HANDLER_PID 2>/dev/null || true
exit 1
