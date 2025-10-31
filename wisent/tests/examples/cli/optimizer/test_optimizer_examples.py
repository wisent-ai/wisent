"""
Tests for optimizer examples (classification and steering).

Validates parameter optimization commands.
"""

import subprocess
import pytest
import tempfile
import os


def test_optimize_classification_parameters():
    """Test optimizing classification parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "optimize-classification",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--limit", "10",
                "--optimization-metric", "f1",
                "--max-time-per-task", "5.0",
                "--layer-range", "2-4",
                "--aggregation-methods", "average", "final",
                "--threshold-range", "0.5", "0.6",
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


def test_optimize_steering_parameters():
    """Test optimizing steering parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "optimize-steering", "comprehensive",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--tasks", "boolq",
                "--methods", "CAA",
                "--limit", "10",
                "--max-time-per-task", "5.0",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
