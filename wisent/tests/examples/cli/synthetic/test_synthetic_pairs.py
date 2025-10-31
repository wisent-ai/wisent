"""
Tests for synthetic pairs generation example.

Validates synthetic contrastive pair creation.
"""

import subprocess
import pytest
import tempfile
import os
import json


def test_create_synthetic_pairs():
    """Test creating synthetic contrastive pairs for a trait."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "synthetic_pairs.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-pairs",
                "--trait", "helpfulness",
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--num-pairs", "5",
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
        
        # Verify JSON format
        with open(output_file, 'r') as f:
            data = json.load(f)
            assert isinstance(data, (list, dict)), "Output should be JSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
