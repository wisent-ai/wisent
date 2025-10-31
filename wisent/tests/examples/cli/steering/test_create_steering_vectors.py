"""
Tests for steering vector creation examples.

Validates creating steering vectors from tasks, synthetic pairs, and activations.
"""

import subprocess
import pytest
import tempfile
import os
import json


def test_create_vector_from_task():
    """Test creating steering vector from lm-eval task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "vector.pt")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-vector-from-task",
                "--task", "boolq",
                "--trait-label", "correctness",
                "--output", output_file,
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
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Vector file not created"


def test_create_vector_from_synthetic():
    """Test creating steering vector from synthetic pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "synthetic_vector.pt")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-vector-from-synthetic",
                "--trait", "helpfulness",
                "--output", output_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--num-pairs", "5",
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
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Vector file not created"


def test_create_vector_from_activations():
    """Test creating steering vector from enriched pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First create pairs with activations
        pairs_file = os.path.join(tmpdir, "pairs.json")
        enriched_file = os.path.join(tmpdir, "enriched_pairs.json")
        vector_file = os.path.join(tmpdir, "vector.pt")
        
        # Create test pairs
        pairs = [
            {
                "prompt": "How should I respond?",
                "positive_response": {"model_response": "Good answer."},
                "negative_response": {"model_response": "Bad answer."},
                "label": "quality"
            },
            {
                "prompt": "What information is best?",
                "positive_response": {"model_response": "Helpful info."},
                "negative_response": {"model_response": "Unhelpful info."},
                "label": "quality"
            }
        ]
        with open(pairs_file, 'w') as f:
            json.dump(pairs, f)
        
        # Enrich with activations
        enrich_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "get-activations",
                pairs_file,
                "--output", enriched_file,
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--layers", "3",
                "--token-aggregation", "average",
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert enrich_result.returncode == 0
        
        # Create steering vector from enriched pairs
        vector_result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "create-steering-vector",
                enriched_file,
                "--output", vector_file,
                "--method", "caa",
                "--normalize",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert vector_result.returncode == 0, f"Vector creation failed: {vector_result.stderr}"
        assert os.path.exists(vector_file), "Vector file not created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
