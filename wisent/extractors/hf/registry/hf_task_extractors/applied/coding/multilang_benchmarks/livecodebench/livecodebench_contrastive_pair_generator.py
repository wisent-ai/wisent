"""
Generate contrastive pairs from LiveCodeBench pre-computed model outputs.

This module creates contrastive pairs by loading existing correct and incorrect
code solutions from the LiveCodeBench dataset's all_outputs.json file.
"""
from __future__ import annotations

from wisent.core.utils.cli.cli_logger import setup_logger
from typing import Any

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from .get_positive_example_livecodebench import (
    get_positive_example,
)
from .get_negative_example_livecodebench import (
    get_negative_example,
)

__all__ = ["generate_livecodebench_pairs"]

log = setup_logger(__name__)


def _load_livecodebench_data(cache_dir: str | None = None) -> list[dict]:
    """
    Load LiveCodeBench data from HuggingFace datasets.

    Args:
        cache_dir: Optional cache directory

    Returns:
        List of problem dictionaries with test cases
    """
    # First, try to load from cached arrow files (most reliable)
    try:
        from datasets import Dataset
        import os

        # Look for cached dataset in standard HuggingFace cache location
        hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
        lcb_cache_base = os.path.join(hf_cache, "livecodebench___code_generation_lite")

        if os.path.exists(lcb_cache_base):
            # Find the latest release directory
            for release_dir in sorted(os.listdir(lcb_cache_base), reverse=True):
                release_path = os.path.join(lcb_cache_base, release_dir)
                if not os.path.isdir(release_path):
                    continue

                # Find version directory
                for version_dir in os.listdir(release_path):
                    version_path = os.path.join(release_path, version_dir)
                    if not os.path.isdir(version_path):
                        continue

                    # Look for arrow files
                    arrow_files = sorted([
                        f for f in os.listdir(version_path)
                        if f.startswith("code_generation_lite-test") and f.endswith(".arrow")
                    ])

                    if arrow_files:
                        all_data = []
                        for arrow_file in arrow_files:
                            arrow_path = os.path.join(version_path, arrow_file)
                            try:
                                ds = Dataset.from_file(arrow_path)
                                all_data.extend([dict(row) for row in ds])
                            except Exception as e:
                                log.warning(f"Could not load arrow file {arrow_file}: {e}")

                        if all_data:
                            log.info(f"Loaded {len(all_data)} problems from cached arrow files")
                            return all_data

    except Exception as e:
        log.warning(f"Could not load from cached arrow files: {e}")

    # Second, try downloading JSONL files directly from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        import json

        # Download the test.jsonl file (contains problems with test cases)
        jsonl_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename="test.jsonl",
            repo_type="dataset",
            cache_dir=cache_dir,
        )

        all_data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if all_data:
            log.info(f"Loaded {len(all_data)} problems from test.jsonl")
            return all_data

    except Exception as e1:
        log.warning(f"Could not load via JSONL download: {e1}")

    # Third, try standard datasets library
    try:
        from datasets import load_dataset

        # Load using standard datasets library (uses cache automatically)
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
        log.info(f"Loaded {len(ds)} problems from livecodebench/code_generation_lite")
        return [dict(row) for row in ds]

    except Exception as e2:
        log.warning(f"Could not load via datasets library: {e2}")

        # Fallback: try to load from the Space's problems.json
        # Note: This fallback does NOT have public_test_cases field
        try:
            from huggingface_hub import hf_hub_download
            import json

            problems_path = hf_hub_download(
                repo_id="livecodebench/code_generation_samples",
                filename="problems.json",
                repo_type="space",
                cache_dir=cache_dir,
            )

            with open(problems_path, "r") as f:
                data = json.load(f)
                log.warning(
                    "Loaded from problems.json which does NOT contain test cases. "
                    "Code evaluation will not work properly."
                )
                return data
        except Exception as e3:
            log.error(f"Could not load problems.json fallback: {e3}")
            return []


def generate_livecodebench_pairs(
    limit: int | None = None,
    cache_dir: str | None = None,
) -> list[ContrastivePair]:
    """
    Generate contrastive pairs from LiveCodeBench dataset.

    This loads pre-computed model outputs (correct and incorrect solutions)
    from the LiveCodeBench dataset and creates contrastive pairs.

    Args:
        limit: Optional maximum number of pairs to generate
        cache_dir: Optional directory to cache downloaded data

    Returns:
        List of ContrastivePair objects with positive (passing) and negative (failing) examples
    """
    try:
        # Load problems.json from the Space to get proper mappings
        from huggingface_hub import hf_hub_download
        import json

        problems_path = hf_hub_download(
            repo_id="livecodebench/code_generation_samples",
            filename="problems.json",
            repo_type="space",
            cache_dir=cache_dir,
        )

        with open(problems_path, "r") as f:
            problems_json = json.load(f)

        max_items = min(limit, len(problems_json)) if limit else len(problems_json)

        pairs: list[ContrastivePair] = []

        log.info(f"Generating contrastive pairs from {max_items} livecodebench problems")

        # Load dataset for test cases
        dataset_data = _load_livecodebench_data(cache_dir)

        # Build question_id -> dataset row mapping for proper matching
        dataset_by_question_id = {}
        for row in dataset_data:
            qid = row.get("question_id")
            if qid:
                dataset_by_question_id[qid] = row

        log.info(f"Loaded {len(dataset_data)} dataset rows, {len(dataset_by_question_id)} with question_id")

        for problem_idx in range(max_items):
            pair = _create_pair_for_problem(problem_idx, problems_json, dataset_data, dataset_by_question_id, cache_dir)
            if pair is not None:
                pairs.append(pair)
                log.debug(f"Created pair {len(pairs)}/{max_items}")

        log.info(f"Generated {len(pairs)} livecodebench contrastive pairs")
        return pairs

    except Exception as exc:
        log.error(f"Error generating livecodebench pairs: {exc}", exc_info=True)
        return []



# Re-exports from split module
from .livecodebench_pair_builder import (
    _create_pair_for_problem,
    _build_test_code,
    _build_stdin_test_code,
    _extract_method_name,
    _build_functional_test_code,
)
