"""
Test for only_generate.sh example.

This test validates that the generate-responses command works correctly
for basic response generation without steering or classification.
"""

import subprocess
import pytest
import tempfile
import os
import json


def test_generate_basic():
    """Test basic response generation from a task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "responses.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-responses",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--task", "boolq",
                "--num-questions", "3",
                "--max-new-tokens", "50",
                "--temperature", "0.7",
                "--top-p", "0.95",
                "--output", output_file,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Output file should exist
        assert os.path.exists(output_file), "Output file was not created"
        
        # Output file should contain valid JSON
        with open(output_file, 'r') as f:
            data = json.load(f)
            assert isinstance(data, (list, dict)), "Output should be JSON"


def test_generate_deterministic():
    """Test deterministic generation with temperature=0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "deterministic.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-responses",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--task", "boolq",
                "--num-questions", "2",
                "--max-new-tokens", "30",
                "--temperature", "0.0",
                "--output", output_file,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Output file should exist
        assert os.path.exists(output_file), "Output file was not created"


def test_generate_creative():
    """Test creative generation with higher temperature."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "creative.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-responses",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--task", "boolq",
                "--num-questions", "2",
                "--max-new-tokens", "40",
                "--temperature", "1.0",
                "--top-p", "0.9",
                "--output", output_file,
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # Should complete without error
        assert result.returncode == 0, f"Command failed with: {result.stderr}"
        
        # Output file should exist
        assert os.path.exists(output_file), "Output file was not created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
