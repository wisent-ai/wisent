"""Benchmark registry for lm-eval-harness tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


__all__ = ["CORE_BENCHMARKS", "BENCHMARKS"]


def _load_core_benchmarks() -> Dict[str, Dict]:
    """Load core benchmarks from JSON file."""
    json_path = Path(__file__).parent / "core_benchmarks.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Core benchmarks JSON file not found: {json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load benchmarks on module import
CORE_BENCHMARKS = _load_core_benchmarks()

# Alias for backward compatibility
BENCHMARKS = CORE_BENCHMARKS
