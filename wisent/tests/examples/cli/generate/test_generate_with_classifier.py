"""
Test for generate_with_classifier.sh example.

This test validates classifier training and inference-only mode
for real-time monitoring during generation.
"""

import subprocess
import pytest
import tempfile
import os


def test_classifier_train_only():
    """Test training a classifier and saving it (train-only mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = os.path.join(tmpdir, "classifier.pt")
        output_dir = os.path.join(tmpdir, "training_logs")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "logistic",
                "--limit", "20",
                "--train-only",
                "--save-classifier", classifier_path,
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
        
        # Classifier file should be saved
        assert os.path.exists(classifier_path), "Classifier was not saved"


def test_classifier_inference_only():
    """Test using a trained classifier during generation (inference-only mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = os.path.join(tmpdir, "classifier.pt")
        training_output = os.path.join(tmpdir, "training")
        inference_output = os.path.join(tmpdir, "inference")
        
        # First train the classifier
        train_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "logistic",
                "--limit", "20",
                "--train-only",
                "--save-classifier", classifier_path,
                "--output", training_output,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"
        assert os.path.exists(classifier_path), "Classifier was not saved"
        
        # Now use it for inference
        inference_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--limit", "10",
                "--inference-only",
                "--load-classifier", classifier_path,
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


def test_classifier_with_threshold():
    """Test classifier with custom detection threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = os.path.join(tmpdir, "classifier.pt")
        training_output = os.path.join(tmpdir, "training")
        inference_output = os.path.join(tmpdir, "inference")
        
        # Train classifier
        train_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "logistic",
                "--limit", "20",
                "--train-only",
                "--save-classifier", classifier_path,
                "--output", training_output,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert train_result.returncode == 0
        
        # Use with custom threshold
        inference_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--limit", "10",
                "--inference-only",
                "--load-classifier", classifier_path,
                "--detection-threshold", "0.7",
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
