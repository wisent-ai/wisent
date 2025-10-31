"""
Tests for contrastive pairs generation examples.

Validates pair generation from tasks and synthetic generation.
"""

import subprocess
import pytest
import tempfile
import os
import json


def test_generate_pairs_from_task():
    """Test generating contrastive pairs from lm-eval task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "pairs.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-pairs-from-task", "boolq",
                "--output", output_file,
                "--limit", "10",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"
        
        # Verify JSON format
        with open(output_file, 'r') as f:
            data = json.load(f)
            assert isinstance(data, (list, dict)), "Output should be JSON"


def test_generate_synthetic_pairs():
    """Test generating synthetic contrastive pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "synthetic_pairs.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-pairs",
                "--trait", "truthfulness",
                "--num-pairs", "5",
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--similarity-threshold", "0.8",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"


def test_generate_synthetic_pairs_different_trait():
    """Test synthetic pair generation with different trait."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "helpfulness_pairs.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-pairs",
                "--trait", "being helpful and informative",
                "--num-pairs", "5",
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
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
