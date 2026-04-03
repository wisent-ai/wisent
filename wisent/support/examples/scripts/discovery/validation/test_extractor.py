"""Test extractor for a single benchmark by running generate-pairs-from-task CLI."""

import json
import os
import subprocess
import sys


def test_extractor(task_name: str, output_file: str, limit: int = 5, timeout: int = 300) -> dict:
    """Run generate-pairs-from-task for a benchmark and verify the output.

    Args:
        task_name: Benchmark task name (e.g. "boolq", "truthfulqa_mc1").
        output_file: Path where the pairs JSON will be written.
        limit: Maximum number of pairs to extract.
        timeout: Max seconds for the CLI command.

    Returns:
        Dict with keys: task, status ("PASS"/"FAIL"), pair_count, detail.
    """
    result = {"task": task_name, "status": "UNKNOWN"}

    proc = subprocess.run(
        [
            sys.executable, "-m", "wisent.core.primitives.model_interface.core.main",
            "generate-pairs-from-task", task_name,
            "--output", output_file,
            "--limit", str(limit),
            "--verbose",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if proc.returncode != 0:
        result["status"] = "FAIL"
        result["detail"] = proc.stderr[:500]
        return result

    if not os.path.exists(output_file):
        result["status"] = "FAIL"
        result["detail"] = "output file not created"
        return result

    with open(output_file) as f:
        data = json.load(f)

    pairs = data["pairs"] if isinstance(data, dict) else data

    if not pairs:
        result["status"] = "FAIL"
        result["detail"] = "zero pairs"
        return result

    result["status"] = "PASS"
    result["pair_count"] = len(pairs)

    return result
