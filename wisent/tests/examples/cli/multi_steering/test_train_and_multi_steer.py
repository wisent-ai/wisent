"""
Test for train_and_multi_steer.sh example.

This test validates training multiple steering vectors and combining them
with different weights during inference using the multi-steer command.
"""

import subprocess
import pytest
import tempfile
import os


def test_train_multiple_vectors_and_combine():
    """Test training multiple vectors and combining them with multi-steer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path = os.path.join(tmpdir, "vector1.pt")
        vector2_path = os.path.join(tmpdir, "vector2.pt")
        combined_path = os.path.join(tmpdir, "combined.pt")
        
        # Train first vector
        train1_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-vector-from-task",
                "--task", "boolq",
                "--trait-label", "truthfulness",
                "--output", vector1_path,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--num-pairs", "10",
                "--layers", "3",
                "--token-aggregation", "average",
                "--method", "caa",
                "--normalize",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert train1_result.returncode == 0, f"Training vector1 failed: {train1_result.stderr}"
        assert os.path.exists(vector1_path), "Vector1 was not saved"
        
        # Train second vector
        train2_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-vector-from-task",
                "--task", "boolq",
                "--trait-label", "helpfulness",
                "--output", vector2_path,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--num-pairs", "10",
                "--layers", "3",
                "--token-aggregation", "average",
                "--method", "caa",
                "--normalize",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert train2_result.returncode == 0, f"Training vector2 failed: {train2_result.stderr}"
        assert os.path.exists(vector2_path), "Vector2 was not saved"
        
        # Combine vectors with equal weights
        combine_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:0.5",
                "--vector", f"{vector2_path}:0.5",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "What is AI?",
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
        
        # Should complete without error
        assert combine_result.returncode == 0, f"Combining failed: {combine_result.stderr}"
        
        # Combined vector should be saved
        assert os.path.exists(combined_path), "Combined vector was not saved"
        
        # Output should contain generated text
        assert len(combine_result.stdout) > 0, "No output generated"


def test_multi_steer_with_different_emphasis():
    """Test combining vectors with different weight emphasis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector1_path = os.path.join(tmpdir, "vector1.pt")
        vector2_path = os.path.join(tmpdir, "vector2.pt")
        
        # Train vectors (abbreviated for test)
        train1_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-vector-from-task",
                "--task", "boolq",
                "--trait-label", "trait1",
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
        
        assert train1_result.returncode == 0
        
        train2_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-vector-from-task",
                "--task", "boolq",
                "--trait-label", "trait2",
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
        
        assert train2_result.returncode == 0
        
        # Emphasize first vector (70% vs 30%)
        combine_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "multi-steer",
                "--vector", f"{vector1_path}:0.7",
                "--vector", f"{vector2_path}:0.3",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layer", "3",
                "--method", "CAA",
                "--prompt", "Explain quantum computing.",
                "--max-new-tokens", "50",
                "--normalize-weights",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Should complete without error
        assert combine_result.returncode == 0, f"Combining failed: {combine_result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
