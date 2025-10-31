"""
Tests for activation extraction examples.

Validates get-activations command with contrastive pairs.
"""

import subprocess
import pytest
import tempfile
import os
import json


def create_test_pairs_file(filepath):
    """Create a simple test pairs JSON file."""
    pairs = [
        {
            "prompt": "What color is the sky?",
            "positive_response": {
                "model_response": "The sky is blue."
            },
            "negative_response": {
                "model_response": "The sky is green."
            },
            "label": "truthfulness"
        },
        {
            "prompt": "What is the chemical formula for water?",
            "positive_response": {
                "model_response": "Water is H2O."
            },
            "negative_response": {
                "model_response": "Water is CO2."
            },
            "label": "truthfulness"
        }
    ]
    with open(filepath, 'w') as f:
        json.dump(pairs, f)


def test_get_activations_from_pairs():
    """Test extracting activations from contrastive pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "pairs.json")
        output_file = os.path.join(tmpdir, "pairs_with_activations.json")
        
        # Create test pairs
        create_test_pairs_file(input_file)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "get-activations",
                input_file,
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layers", "3",
                "--token-aggregation", "average",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"


def test_get_activations_multiple_layers():
    """Test extracting activations from multiple layers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "pairs.json")
        output_file = os.path.join(tmpdir, "multilayer_activations.json")
        
        create_test_pairs_file(input_file)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "get-activations",
                input_file,
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layers", "2,3,4",
                "--token-aggregation", "average",
                "--device", "cpu",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"


def test_get_activations_different_aggregation():
    """Test different token aggregation strategies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "pairs.json")
        output_file = os.path.join(tmpdir, "final_token_activations.json")
        
        create_test_pairs_file(input_file)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "get-activations",
                input_file,
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layers", "3",
                "--token-aggregation", "final",
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
