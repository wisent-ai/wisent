"""Pull pre-computed full_benchmarks pickles from the wisent-ai HF dataset.

Lazy alternative to `download_full_benchmarks.py` which rebuilds every
benchmark from scratch via lm-eval-harness. This helper downloads the
precomputed classifier artifacts (80 .pkl + 74 .json) from HuggingFace
directly, much faster and avoiding dataset gating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

HF_REPO_ID = "wisent-ai/classifier-benchmarks"


def fetch_full_benchmarks(
    target_dir: Optional[Path] = None,
    revision: str = "main",
    force: bool = False,
) -> Path:
    """Download the classifier-benchmarks dataset into target_dir.

    Args:
        target_dir: Where to put the files. Defaults to
            ``~/.cache/wisent/classifier-benchmarks``.
        revision: HF revision (branch, tag, or commit sha) to fetch.
        force: Re-download even if target_dir already has files.

    Returns:
        Path to the directory holding the downloaded data/ and metadata/
        subfolders.
    """
    from huggingface_hub import snapshot_download

    if target_dir is None:
        target_dir = Path.home() / ".cache" / "wisent" / "classifier-benchmarks"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(target_dir.rglob("*.pkl")):
        return target_dir

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir


def ensure_full_benchmarks() -> Path:
    """Return the on-disk path to the full_benchmarks cache, downloading if missing.

    Safe to call multiple times; subsequent calls are free after the first.
    """
    return fetch_full_benchmarks(force=False)
