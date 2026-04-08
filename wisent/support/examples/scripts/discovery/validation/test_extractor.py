"""Test extractor for a single benchmark by running generate-pairs-from-task CLI."""

import json
import os
import subprocess
import sys


def test_extractor(task_name: str, output_file: str, limit: int = 5) -> dict:
    """Run generate-pairs-from-task for a benchmark and verify the output.

    Args:
        task_name: Benchmark task name (e.g. "boolq", "truthfulqa_mc1").
        output_file: Path where the pairs JSON will be written.
        limit: Maximum number of pairs to extract.

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
            "--allow-subtasks",
            "--verbose",
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        result["status"] = "FAIL"
        # Capture more stderr so TaskSelector log noise doesn't drown out the real
        # exception traceback (the first ~500 chars of stderr are often warning logs).
        result["detail"] = proc.stderr[-3000:] if len(proc.stderr) > 3000 else proc.stderr
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
