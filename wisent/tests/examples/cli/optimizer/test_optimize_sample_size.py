"""
Test for optimize_sample_size.sh example.

This test validates that the optimize-sample-size command works correctly
with minimal parameters to verify the example script would work.
"""

import subprocess
import pytest
import tempfile
import os


def test_optimize_sample_size_basic():
    """Test basic sample size optimization command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "optimize-sample-size",
                "meta-llama/Llama-3.2-1B-Instruct",  # Small model for testing
                "--task", "boolq",
                "--layer", "3",
                "--token-aggregation", "average",
                "--sample-sizes", "5", "10",
                "--test-size", "20",
                "--limit", "30",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Should mention sample sizes in output
        assert "5" in result.stdout or "5" in result.stderr
        assert "10" in result.stdout or "10" in result.stderr


def test_optimize_sample_size_with_steering():
    """Test sample size optimization in steering mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "optimize-sample-size",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--task", "boolq",
                "--layer", "3",
                "--token-aggregation", "final",
                "--steering-mode",
                "--steering-method", "CAA",
                "--steering-strength", "1.0",
                "--sample-sizes", "5", "10",
                "--test-size", "15",
                "--limit", "25",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Should mention steering in output
        assert "steering" in result.stdout.lower() or "steering" in result.stderr.lower()


def test_optimize_sample_size_custom_threshold():
    """Test sample size optimization with custom threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "optimize-sample-size",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--task", "boolq",
                "--layer", "3",
                "--token-aggregation", "max",
                "--threshold", "0.6",
                "--sample-sizes", "5", "10",
                "--test-size", "15",
                "--limit", "25",
                "--device", "cpu",
                "--seed", "123",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
