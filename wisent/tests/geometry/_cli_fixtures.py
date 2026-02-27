"""Shared fixtures and helpers for CLI command tests."""

import subprocess
import json


# Configuration
MODEL = "gpt2"
TASK = "boolq"
TIMEOUT = 600


def run_wisent_command(
    args: list, timeout: int = TIMEOUT
) -> subprocess.CompletedProcess:
    """Run a wisent CLI command and return the result."""
    cmd = ["python", "-m", "wisent.core.main"] + args
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )


def make_test_pairs():
    """Return list of test pair dicts."""
    return [
        {
            "prompt": "What color is the sky?",
            "positive_response": {
                "model_response": "The sky is blue during a clear day."
            },
            "negative_response": {
                "model_response": "The sky is green with purple stripes."
            },
            "label": "truthfulness",
        },
        {
            "prompt": "What is 2 + 2?",
            "positive_response": {
                "model_response": "2 + 2 equals 4."
            },
            "negative_response": {
                "model_response": "2 + 2 equals 7."
            },
            "label": "truthfulness",
        },
        {
            "prompt": "What is the capital of France?",
            "positive_response": {
                "model_response": "Paris is the capital of France."
            },
            "negative_response": {
                "model_response": "Moscow is the capital of France."
            },
            "label": "truthfulness",
        },
        {
            "prompt": "How many days in a week?",
            "positive_response": {
                "model_response": "There are 7 days in a week."
            },
            "negative_response": {
                "model_response": "There are 12 days in a week."
            },
            "label": "truthfulness",
        },
    ]


def make_test_responses():
    """Return list of test response dicts."""
    return [
        {
            "question": "Is the sky blue?",
            "response": "Yes",
            "expected": "Yes",
            "choices": ["Yes", "No"],
        },
        {
            "question": "Is water wet?",
            "response": "Yes",
            "expected": "Yes",
            "choices": ["Yes", "No"],
        },
    ]


def run_ok(args, **kwargs):
    """Run command and assert success."""
    result = run_wisent_command(args, **kwargs)
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    return result
