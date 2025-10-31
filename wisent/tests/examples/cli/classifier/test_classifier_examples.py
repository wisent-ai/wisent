"""
Tests for classifier examples.

Validates classifier training, loading, and evaluation workflows.
"""

import subprocess
import pytest
import tempfile
import os


def test_train_classifier_and_save():
    """Test training a classifier and saving it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = os.path.join(tmpdir, "classifier.pt")
        output_dir = os.path.join(tmpdir, "training")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "logistic",
                "--limit", "20",
                "--save-classifier", classifier_path,
                "--output", output_dir,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(classifier_path), "Classifier not saved"


def test_use_pretrained_classifier():
    """Test loading and using a pretrained classifier."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = os.path.join(tmpdir, "classifier.pt")
        training_output = os.path.join(tmpdir, "training")
        inference_output = os.path.join(tmpdir, "inference")
        
        # Train classifier first
        train_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "logistic",
                "--limit", "20",
                "--save-classifier", classifier_path,
                "--output", training_output,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert train_result.returncode == 0
        
        # Use pretrained classifier
        inference_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--load-classifier", classifier_path,
                "--inference-only",
                "--testing-limit", "10",
                "--output", inference_output,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        assert inference_result.returncode == 0, f"Inference failed: {inference_result.stderr}"


def test_run_and_evaluate_on_benchmark():
    """Test training and evaluating classifier on benchmark."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "benchmark")
        report_file = os.path.join(output_dir, "report.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "logistic",
                "--training-limit", "20",
                "--testing-limit", "10",
                "--token-aggregation", "average",
                "--detection-threshold", "0.6",
                "--output", output_dir,
                "--evaluation-report", report_file,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


def test_classifier_with_mlp():
    """Test training MLP classifier (not just logistic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = os.path.join(tmpdir, "mlp_classifier.pt")
        output_dir = os.path.join(tmpdir, "training")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "tasks", "boolq",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--classifier-type", "mlp",
                "--limit", "20",
                "--save-classifier", classifier_path,
                "--output", output_dir,
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(classifier_path), "MLP classifier not saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
