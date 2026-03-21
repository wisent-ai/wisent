"""Storage-first loading and upload for contrastive pair texts.

Implements the cascade: local cache -> HuggingFace -> Supabase -> generate.
Converts between stored dict format and ContrastivePair objects.
Uploads newly generated pairs to HF for future reuse.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair

_LOG = setup_logger(__name__)


def pairs_from_texts(
    pair_texts: Dict[int, Dict[str, str]],
) -> List["ContrastivePair"]:
    """Convert stored pair texts dict to ContrastivePair objects.

    Args:
        pair_texts: Dict mapping pair_id to {prompt, positive, negative}

    Returns:
        List of ContrastivePair objects sorted by pair_id
    """
    from wisent.core.primitives.contrastive_pairs.core.pair import (
        ContrastivePair,
    )
    from wisent.core.primitives.contrastive_pairs.core.io.response import (
        PositiveResponse, NegativeResponse,
    )
    pairs = []
    for _pid, entry in sorted(pair_texts.items()):
        pairs.append(ContrastivePair(
            prompt=entry["prompt"],
            positive_response=PositiveResponse(
                model_response=entry["positive"],
            ),
            negative_response=NegativeResponse(
                model_response=entry["negative"],
            ),
        ))
    return pairs


def try_load_from_storage(
    task_name: str, limit: int | None,
) -> list["ContrastivePair"] | None:
    """Try loading pair texts from cache, HF, or Supabase.

    Cascade order:
        - Local cache (~/.wisent_cache)
        - HuggingFace Hub (wisent-ai/activations)
        - Supabase database

    Args:
        task_name: Benchmark/task name
        limit: Optional max pairs to return

    Returns:
        List of ContrastivePair if found, None otherwise
    """
    log = bind(_LOG, task=task_name)
    effective_limit = limit if limit else None

    # Local cache
    from wisent.core.reading.modules.utilities.data.cache import (
        load_pair_texts_cache,
    )
    cached = load_pair_texts_cache(task_name, limit=effective_limit)
    if cached:
        log.info(
            "Loaded pairs from local cache",
            extra={"count": len(cached)},
        )
        return pairs_from_texts(cached)

    # HuggingFace
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
            load_pair_texts_from_hf,
        )
        hf_pairs = load_pair_texts_from_hf(
            task_name, limit=effective_limit,
        )
        if hf_pairs:
            log.info(
                "Loaded pairs from HuggingFace",
                extra={"count": len(hf_pairs)},
            )
            return pairs_from_texts(hf_pairs)
    except Exception as exc:
        log.debug(f"HF load failed: {exc}")

    # Supabase
    try:
        from wisent.core.reading.modules.utilities.data.database_loaders import (
            load_pair_texts_from_database,
        )
        db_pairs = load_pair_texts_from_database(
            task_name, limit=effective_limit,
        )
        if db_pairs:
            log.info(
                "Loaded pairs from Supabase",
                extra={"count": len(db_pairs)},
            )
            return pairs_from_texts(db_pairs)
    except Exception as exc:
        log.debug(f"Supabase load failed: {exc}")

    return None


def upload_pairs_to_hf(
    task_name: str, pairs: list["ContrastivePair"],
) -> None:
    """Upload generated pairs to HF for future reuse.

    Converts ContrastivePair objects to the stored dict format
    and uploads via hf_writers. Silently catches errors since
    this is a best-effort cache population.

    Args:
        task_name: Benchmark/task name
        pairs: List of ContrastivePair objects to upload
    """
    log = bind(_LOG, task=task_name)
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            upload_pair_texts,
        )
        pair_texts = {}
        for i, p in enumerate(pairs):
            entry = {
                "prompt": p.prompt,
                "positive": p.positive_response.model_response,
                "negative": p.negative_response.model_response,
            }
            if p.metadata:
                entry["metadata"] = p.metadata
            pair_texts[i] = entry
        upload_pair_texts(task_name, pair_texts)
        log.info(
            "Uploaded pairs to HuggingFace",
            extra={"count": len(pairs)},
        )
    except Exception as exc:
        log.warning(f"Failed to upload pairs to HF: {exc}")
