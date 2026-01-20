"""
Storage utilities for contrastive pairs.

Provides high-level functions for saving and loading contrastive pairs
from the centralized data directory structure.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.contrastive_pairs.core.serialization import (
    save_contrastive_pair_set,
    load_contrastive_pair_set,
)

from . import (
    PAIRS_DIR,
    PERSONALIZATION_DIR,
    SYNTHETIC_DIR,
    BENCHMARKS_DIR,
    TRAIT_DIRS,
    get_trait_dir,
)

__all__ = [
    "save_personalization_pairs",
    "load_personalization_pairs",
    "save_synthetic_pairs",
    "load_synthetic_pairs",
    "save_benchmark_pairs",
    "load_benchmark_pairs",
    "list_stored_pairs",
    "get_pair_count",
]


def _generate_filename(
    name: str,
    model: str | None = None,
    include_timestamp: bool = True,
) -> str:
    """Generate a filename for storing pairs.

    Args:
        name: Base name for the file (e.g., trait name or benchmark name)
        model: Optional model name to include
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Generated filename with .json extension
    """
    parts = [name.lower().replace(" ", "_").replace("-", "_")]
    if model:
        model_clean = model.replace("/", "_").replace("-", "_").lower()
        parts.append(model_clean)
    if include_timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts) + ".json"


def save_personalization_pairs(
    pairs: ContrastivePairSet | list[ContrastivePair] | list[dict[str, Any]],
    trait: str,
    model: str | None = None,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save personalization contrastive pairs for a specific trait.

    Args:
        pairs: ContrastivePairSet, list of ContrastivePair objects, or list of dicts
        trait: Personality trait name (e.g., "british", "evil", "flirty", "left_wing")
        model: Optional model name used to generate pairs
        filename: Optional custom filename (auto-generated if not provided)
        metadata: Optional metadata to include in the file

    Returns:
        Path to the saved file
    """
    trait_dir = get_trait_dir(trait)
    trait_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = _generate_filename(trait, model, include_timestamp=True)

    filepath = trait_dir / filename

    # Convert to ContrastivePairSet if needed
    if isinstance(pairs, ContrastivePairSet):
        pair_set = pairs
    elif isinstance(pairs, list):
        if len(pairs) > 0 and isinstance(pairs[0], dict):
            pair_list = [ContrastivePair.from_dict(p) for p in pairs]
        else:
            pair_list = pairs
        pair_set = ContrastivePairSet(
            name=f"{trait}_personalization",
            pairs=pair_list,
            task_type="personalization",
        )
    else:
        raise TypeError(f"Expected ContrastivePairSet or list, got {type(pairs)}")

    save_contrastive_pair_set(pair_set, filepath)

    # If metadata provided, save it alongside
    if metadata:
        meta_filepath = filepath.with_suffix(".meta.json")
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_personalization_pairs(
    trait: str,
    filename: str | None = None,
    return_backend: str = "torch",
) -> ContrastivePairSet:
    """Load personalization contrastive pairs for a specific trait.

    Args:
        trait: Personality trait name
        filename: Specific filename to load (loads latest if not provided)
        return_backend: Backend for activations ('torch', 'numpy', or 'list')

    Returns:
        ContrastivePairSet with the loaded pairs
    """
    trait_dir = get_trait_dir(trait)

    if filename:
        filepath = trait_dir / filename
    else:
        # Load the most recent file
        json_files = sorted(trait_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        json_files = [f for f in json_files if not f.name.endswith(".meta.json")]
        if not json_files:
            raise FileNotFoundError(f"No pairs found for trait '{trait}' in {trait_dir}")
        filepath = json_files[-1]

    return load_contrastive_pair_set(filepath, return_backend=return_backend)


def save_synthetic_pairs(
    pairs: ContrastivePairSet | list[ContrastivePair] | list[dict[str, Any]],
    name: str,
    model: str | None = None,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save synthetically generated contrastive pairs.

    Args:
        pairs: ContrastivePairSet, list of ContrastivePair objects, or list of dicts
        name: Name/category for the pairs
        model: Optional model name used to generate pairs
        filename: Optional custom filename
        metadata: Optional metadata to include

    Returns:
        Path to the saved file
    """
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = _generate_filename(name, model, include_timestamp=True)

    filepath = SYNTHETIC_DIR / filename

    if isinstance(pairs, ContrastivePairSet):
        pair_set = pairs
    elif isinstance(pairs, list):
        if len(pairs) > 0 and isinstance(pairs[0], dict):
            pair_list = [ContrastivePair.from_dict(p) for p in pairs]
        else:
            pair_list = pairs
        pair_set = ContrastivePairSet(
            name=f"{name}_synthetic",
            pairs=pair_list,
            task_type="synthetic",
        )
    else:
        raise TypeError(f"Expected ContrastivePairSet or list, got {type(pairs)}")

    save_contrastive_pair_set(pair_set, filepath)

    if metadata:
        meta_filepath = filepath.with_suffix(".meta.json")
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_synthetic_pairs(
    name: str | None = None,
    filename: str | None = None,
    return_backend: str = "torch",
) -> ContrastivePairSet:
    """Load synthetically generated contrastive pairs.

    Args:
        name: Name/category pattern to filter by
        filename: Specific filename to load
        return_backend: Backend for activations

    Returns:
        ContrastivePairSet with the loaded pairs
    """
    if filename:
        filepath = SYNTHETIC_DIR / filename
    elif name:
        pattern = f"{name.lower().replace(' ', '_')}*.json"
        json_files = sorted(SYNTHETIC_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        json_files = [f for f in json_files if not f.name.endswith(".meta.json")]
        if not json_files:
            raise FileNotFoundError(f"No synthetic pairs found matching '{name}'")
        filepath = json_files[-1]
    else:
        json_files = sorted(SYNTHETIC_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        json_files = [f for f in json_files if not f.name.endswith(".meta.json")]
        if not json_files:
            raise FileNotFoundError("No synthetic pairs found")
        filepath = json_files[-1]

    return load_contrastive_pair_set(filepath, return_backend=return_backend)


def save_benchmark_pairs(
    pairs: ContrastivePairSet | list[ContrastivePair] | list[dict[str, Any]],
    benchmark: str,
    model: str | None = None,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save benchmark-extracted contrastive pairs.

    Args:
        pairs: ContrastivePairSet, list of ContrastivePair objects, or list of dicts
        benchmark: Benchmark name (e.g., "truthfulqa", "hellaswag")
        model: Optional model name
        filename: Optional custom filename
        metadata: Optional metadata to include

    Returns:
        Path to the saved file
    """
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = _generate_filename(benchmark, model, include_timestamp=True)

    filepath = BENCHMARKS_DIR / filename

    if isinstance(pairs, ContrastivePairSet):
        pair_set = pairs
    elif isinstance(pairs, list):
        if len(pairs) > 0 and isinstance(pairs[0], dict):
            pair_list = [ContrastivePair.from_dict(p) for p in pairs]
        else:
            pair_list = pairs
        pair_set = ContrastivePairSet(
            name=f"{benchmark}_benchmark",
            pairs=pair_list,
            task_type="benchmark",
        )
    else:
        raise TypeError(f"Expected ContrastivePairSet or list, got {type(pairs)}")

    save_contrastive_pair_set(pair_set, filepath)

    if metadata:
        meta_filepath = filepath.with_suffix(".meta.json")
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_benchmark_pairs(
    benchmark: str | None = None,
    filename: str | None = None,
    return_backend: str = "torch",
) -> ContrastivePairSet:
    """Load benchmark-extracted contrastive pairs.

    Args:
        benchmark: Benchmark name to filter by
        filename: Specific filename to load
        return_backend: Backend for activations

    Returns:
        ContrastivePairSet with the loaded pairs
    """
    if filename:
        filepath = BENCHMARKS_DIR / filename
    elif benchmark:
        pattern = f"{benchmark.lower().replace(' ', '_')}*.json"
        json_files = sorted(BENCHMARKS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        json_files = [f for f in json_files if not f.name.endswith(".meta.json")]
        if not json_files:
            raise FileNotFoundError(f"No benchmark pairs found for '{benchmark}'")
        filepath = json_files[-1]
    else:
        json_files = sorted(BENCHMARKS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        json_files = [f for f in json_files if not f.name.endswith(".meta.json")]
        if not json_files:
            raise FileNotFoundError("No benchmark pairs found")
        filepath = json_files[-1]

    return load_contrastive_pair_set(filepath, return_backend=return_backend)


def list_stored_pairs(category: str | None = None) -> dict[str, list[Path]]:
    """List all stored contrastive pair files.

    Args:
        category: Filter by category ('personalization', 'synthetic', 'benchmarks')
                  If None, returns all categories.

    Returns:
        Dictionary mapping category/trait names to lists of file paths
    """
    result: dict[str, list[Path]] = {}

    if category is None or category == "personalization":
        for trait, trait_dir in TRAIT_DIRS.items():
            if trait_dir.exists():
                files = [f for f in trait_dir.glob("*.json") if not f.name.endswith(".meta.json")]
                if files:
                    result[f"personalization/{trait}"] = sorted(files)

    if category is None or category == "synthetic":
        if SYNTHETIC_DIR.exists():
            files = [f for f in SYNTHETIC_DIR.glob("*.json") if not f.name.endswith(".meta.json")]
            if files:
                result["synthetic"] = sorted(files)

    if category is None or category == "benchmarks":
        if BENCHMARKS_DIR.exists():
            files = [f for f in BENCHMARKS_DIR.glob("*.json") if not f.name.endswith(".meta.json")]
            if files:
                result["benchmarks"] = sorted(files)

    return result


def get_pair_count(category: str | None = None) -> dict[str, int]:
    """Get count of pairs in each category/trait.

    Args:
        category: Filter by category or None for all

    Returns:
        Dictionary mapping category/trait to pair count
    """
    stored = list_stored_pairs(category)
    counts: dict[str, int] = {}

    for key, files in stored.items():
        total = 0
        for filepath in files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    if "pairs" in data:
                        total += len(data["pairs"])
            except (json.JSONDecodeError, KeyError):
                pass
        counts[key] = total

    return counts
