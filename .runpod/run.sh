#!/bin/bash
# Wisent - Runpod Startup Script
# Starts base image services (Jupyter/SSH) then sets up workspace

set -e

echo "=============================================="
echo "  Wisent - Runpod Pod Template"
echo "=============================================="

# Start base image services (Jupyter/SSH) in background
# The base image's /start.sh handles Jupyter and SSH automatically
/start.sh &

# Wait for services to initialize
sleep 3

# Create workspace directories (volume mount overwrites build-time files)
mkdir -p /workspace/notebooks /workspace/outputs /workspace/models

# Copy notebooks if they don't exist yet
if [ ! -f "/workspace/notebooks/hallucination_guard.ipynb" ]; then
    echo "Copying example notebooks to /workspace/notebooks..."
    cp -r /wisent/wisent/examples/notebooks/* /workspace/notebooks/ 2>/dev/null || true
fi

echo ""
echo "Wisent is ready!"
echo ""
echo "Available notebooks in /workspace/notebooks:"
ls -1 /workspace/notebooks/*.ipynb 2>/dev/null || echo "  (none found)"
echo ""
echo "Access JupyterLab via Runpod console 'Connect' button"
echo "=============================================="

# Keep container running and wait for background processes
wait
