"""
Test for generate_with_steering.sh example.

This test validates steering vector training and inference-only mode
for controlling model behavior during generation.
"""

import subprocess
import pytest
import tempfile
import os


def test_steering_vector_train_only():
    """Test training a steering vector and saving it (train-only mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_path = os.path.join(tmpdir, "vector.pt")
        output_dir = os.path.join(tmpdir, "training_logs")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--steering-method", "CAA",
                "--limit", "20",
                "--train-only",
                "--save-steering-vector", vector_path,
                "--output", output_dir,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Vector file should be saved
        assert os.path.exists(vector_path), "Steering vector was not saved"


def test_steering_vector_inference_only():
    """Test using a trained steering vector during generation (inference-only mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_path = os.path.join(tmpdir, "vector.pt")
        training_output = os.path.join(tmpdir, "training")
        inference_output = os.path.join(tmpdir, "inference")
        
        # First train the vector
        train_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--steering-method", "CAA",
                "--limit", "20",
                "--train-only",
                "--save-steering-vector", vector_path,
                "--output", training_output,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"
        assert os.path.exists(vector_path), "Vector was not saved"
        
        # Now use it for inference
        inference_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--steering-method", "CAA",
                "--steering-strength", "1.5",
                "--limit", "10",
                "--inference-only",
                "--load-steering-vector", vector_path,
                "--output", inference_output,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should complete without error
        assert inference_result.returncode == 0, f"Inference failed: {inference_result.stderr}"


def test_steering_with_caa_l2():
    """Test steering with CAA method using strong steering strength."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_path = os.path.join(tmpdir, "vector.pt")
        training_output = os.path.join(tmpdir, "training")
        inference_output = os.path.join(tmpdir, "inference")

        # Train with CAA
        train_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--steering-method", "CAA",
                "--limit", "20",
                "--train-only",
                "--save-steering-vector", vector_path,
                "--output", training_output,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert train_result.returncode == 0

        # Use with stronger steering
        inference_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--steering-method", "CAA",
                "--steering-strength", "2.0",
                "--limit", "10",
                "--inference-only",
                "--load-steering-vector", vector_path,
                "--output", inference_output,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Should complete without error
        assert inference_result.returncode == 0, f"Inference failed: {inference_result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
