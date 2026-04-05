"""Test evaluator for a single benchmark by running evaluate-responses CLI."""

import json
import os
import subprocess
import sys
import tempfile


def test_evaluator(task_name: str, responses: list[dict], model_name: str | None = None) -> dict:
    """Run evaluate-responses for a benchmark and verify the output.

    Args:
        task_name: Benchmark task name (e.g. "boolq", "truthfulqa_mc1").
        responses: List of response dicts with keys: question, response, expected, choices.
        model_name: HuggingFace model name for log-likelihood evaluation.

    Returns:
        Dict with keys: task, status ("PASS"/"FAIL"), detail.
    """
    result = {"task": task_name, "status": "UNKNOWN"}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "responses.json")
        output_file = os.path.join(tmpdir, "evaluation.json")

        with open(input_file, "w") as f:
            json.dump(responses, f)

        proc = subprocess.run(
            [
                sys.executable, "-m", "wisent.core.primitives.model_interface.core.main",
                "evaluate-responses",
                "--input", input_file,
                "--output", output_file,
                "--task", task_name,
                "--subprocess-timeout", "120",
                "--personalization-good-threshold", "50",
                "--fast-diversity-seed", "42",
                "--diversity-max-sample-size", "100",
                "--min-sentence-length", "5",
                "--nonsense-min-tokens", "3",
                "--quality-min-response-length", "10",
                "--quality-repetition-ratio-threshold", "0.5",
                "--quality-bigram-repeat-threshold", "3",
                "--quality-bigram-repeat-penalty", "0.1",
                "--quality-special-char-ratio-threshold", "0.3",
                "--quality-special-char-penalty", "0.1",
                "--quality-char-repeat-count", "4",
                "--quality-char-repeat-penalty", "0.1",
                "--f1-threshold", "0.5",
                "--generation-embedding-weight", "0.5",
                "--generation-nli-weight", "0.5",
                "--personalization-difference-weight", "0.33",
                "--personalization-quality-weight", "0.33",
                "--personalization-alignment-weight", "0.34",
                "--verbose",
            ]
            + (["--model", model_name] if model_name else []),
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            result["status"] = "FAIL"
            result["detail"] = proc.stderr[:2000]
            return result

        if not os.path.exists(output_file):
            result["status"] = "FAIL"
            result["detail"] = "output file not created"
            return result

        with open(output_file) as f:
            data = json.load(f)

        result["evaluation"] = data

        num_evaluated = data.get("num_evaluated", 0)
        num_total = data.get("num_total", 0)
        num_model_required = data.get("num_model_required", 0)

        if num_evaluated == 0 and num_model_required > 0:
            result["status"] = "SKIP_NO_MODEL"
            result["detail"] = f"evaluator requires model (0/{num_total} evaluated, {num_model_required} need model)"
            return result

        if num_evaluated == 0:
            result["status"] = "FAIL"
            result["detail"] = f"0/{num_total} responses evaluated"
            return result

        result["status"] = "PASS"

    return result
