from __future__ import annotations

import gzip
import io
import json
from typing import Any

import requests

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger

__all__ = ["Anagrams2Extractor"]
_LOG = setup_logger(__name__)

# Raw GitHub URL for the mid_word_2_anagrams dataset (GPT-3 data release)
_ANAGRAMS2_URL = (
    "https://raw.githubusercontent.com/openai/gpt-3/master/data/mid_word_2_anagrams.jsonl.gz"
)

task_names = ("anagrams2",)


class Anagrams2Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Anagrams2 benchmark.

    Downloads mid_word_2_anagrams data directly from the GPT-3 GitHub
    repository to avoid the lm_eval trust_remote_code issue with
    EleutherAI/unscramble on newer datasets library versions.
    """

    evaluator_name = "exact_match"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Load anagrams2 docs from GitHub and build contrastive pairs."""
        max_items = self._normalize_limit(limit)
        docs = self._load_docs_from_github()

        if not docs:
            _LOG.warning("No docs loaded for anagrams2")
            return []

        if len(docs) < 2:
            _LOG.warning("Not enough docs to create pairs", extra={"doc_count": len(docs)})
            return []

        pairs: list[ContrastivePair] = []
        for i, doc in enumerate(docs):
            context = doc.get("context", "").strip()
            completion = doc.get("completion", "").strip()
            if not context or not completion:
                continue

            incorrect_completion = docs[(i + 1) % len(docs)].get("completion", "").strip()
            if not incorrect_completion or incorrect_completion == completion:
                continue

            pair = ContrastivePair(
                prompt=context,
                positive_response=PositiveResponse(model_response=completion),
                negative_response=NegativeResponse(model_response=incorrect_completion),
                label="anagrams2",
            )
            pairs.append(pair)

            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            _LOG.warning("No valid anagrams2 pairs extracted")
        return pairs

    def _load_docs_from_github(self) -> list[dict[str, Any]]:
        """Download and parse the mid_word_2_anagrams JSONL file from GitHub."""
        try:
            resp = requests.get(_ANAGRAMS2_URL, timeout=30)
            resp.raise_for_status()
            with gzip.open(io.BytesIO(resp.content)) as f:
                lines = f.read().decode("utf-8").splitlines()
            docs = [json.loads(line) for line in lines if line.strip()]
            _LOG.info(f"Loaded {len(docs)} anagrams2 docs from GitHub")
            return docs
        except Exception as exc:
            _LOG.error(f"Failed to load anagrams2 data from GitHub: {exc}")
            return []
