"""
Tests for evaluation examples.

Validates response evaluation and personalization assessment.
"""

import subprocess
import pytest
import tempfile
import os
import json


def create_test_responses_file(filepath):
    """Create a test responses JSON file."""
    responses = [
        {
            "question": "What is 2+2?",
            "response": "4",
            "expected": "4",
            "choices": ["3", "4", "5", "6"]
        },
        {
            "question": "What is the capital of France?",
            "response": "Paris",
            "expected": "Paris",
            "choices": ["London", "Paris", "Berlin", "Rome"]
        }
    ]
    with open(filepath, 'w') as f:
        json.dump(responses, f)


def test_generate_responses_from_task():
    """Test generating responses from a task."""
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
                "--device", "cpu",
                "--output", output_file,
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"


def test_evaluate_generated_responses():
    """Test evaluating generated responses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "responses.json")
        output_file = os.path.join(tmpdir, "evaluation.json")

        # Create test responses
        create_test_responses_file(input_file)

        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "evaluate-responses",
                "--input", input_file,
                "--output", output_file,
                "--task", "boolq",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"


def test_evaluate_personalization():
    """Test evaluating personalization responses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "personalization.json")
        output_file = os.path.join(tmpdir, "evaluation.json")
        
        # Create test responses
        responses = [
            {"question": "Test Q", "response": "Test A", "trait": "helpful"}
        ]
        with open(input_file, 'w') as f:
            json.dump(responses, f)
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "evaluate-responses",
                "--input", input_file,
                "--output", output_file,
                "--task", "personalization",
                "--trait", "helpful",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
