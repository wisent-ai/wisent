"""
Test for multi_steer_with_different_parameters.sh example.

This test validates different parameter combinations for multi-steering:
normalized vs unnormalized weights, target norm scaling, and different methods.
"""

import subprocess
import pytest
import tempfile
import os


def setup_test_vectors(tmpdir):
    """Helper function to create test vectors for multi-steering tests."""
    vector1_path = os.path.join(tmpdir, "vector1.pt")
    vector2_path = os.path.join(tmpdir, "vector2.pt")
    
    # Train first vector
    result1 = subprocess.run(
        [
            "python", "-m", "wisent.core.main", "generate-vector-from-task",
            "--task", "boolq",
            "--trait-label", "test1",
            "--output", vector1_path,
            "--model", "meta-llama/Llama-3.2-1B-Instruct",
            "--num-pairs", "10",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize",
            "--device", "cpu"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert result1.returncode == 0, f"Failed to create vector1: {result1.stderr}"
    
    # Train second vector
    result2 = subprocess.run(
        [
            "python", "-m", "wisent.core.main", "generate-vector-from-task",
            "--task", "boolq",
            "--trait-label", "test2",
            "--output", vector2_path,
            "--model", "meta-llama/Llama-3.2-1B-Instruct",
            "--num-pairs", "10",
            "--layers", "3",
            "--token-aggregation", "average",
            "--method", "caa",
            "--normalize",
            "--device", "cpu"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert result2.returncode == 0, f"Failed to create vector2: {result2.stderr}"
    
    return vector1_path, vector2_path


def test_normalized_weights():
    """Test multi-steer with normalized weights (sum to 1.0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path, vector2_path = setup_test_vectors(tmpdir)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:0.6",
                "--vector", f"{vector2_path}:0.4",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain AI.",
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


def test_unnormalized_weights():
    """Test multi-steer with unnormalized weights for stronger effect."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path, vector2_path = setup_test_vectors(tmpdir)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:2.0",
                "--vector", f"{vector2_path}:1.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain AI.",
                "--max-new-tokens", "50",
                "--allow-unnormalized",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


def test_target_norm_scaling():
    """Test multi-steer with target norm scaling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path, vector2_path = setup_test_vectors(tmpdir)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:0.5",
                "--vector", f"{vector2_path}:0.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain AI.",
                "--max-new-tokens", "50",
                "--target-norm", "10.0",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


def test_subtle_steering_with_small_norm():
    """Test subtle steering with small target norm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path, vector2_path = setup_test_vectors(tmpdir)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:0.5",
                "--vector", f"{vector2_path}:0.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain AI.",
                "--max-new-tokens", "50",
                "--target-norm", "2.0",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


def test_save_combined_vector():
    """Test saving combined vector for reuse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path, vector2_path = setup_test_vectors(tmpdir)
        combined_path = os.path.join(tmpdir, "combined.pt")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:0.5",
                "--vector", f"{vector2_path}:0.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain AI.",
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--save-combined", combined_path,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(combined_path), "Combined vector was not saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
