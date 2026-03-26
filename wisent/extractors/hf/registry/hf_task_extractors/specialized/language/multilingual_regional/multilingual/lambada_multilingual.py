from __future__ import annotations

import random
from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

class LambadaMultilingualExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for LAMBADA Multilingual - Word prediction benchmark.

    Dataset: EleutherAI/lambada_openai on HuggingFace

    LAMBADA tests language models on their ability to predict the last word
    of a passage that requires broad context to resolve.
    Multilingual variants available for multiple languages.
    """

    evaluator_name = "lambada_multilingual"

    def __init__(self, language: str | None = None):
        """
        Initialize LAMBADA Multilingual extractor.

        Args:
            language: Optional language filter (e.g., 'de', 'fr', 'it', 'es')
        """
        super().__init__()
        self.language = language or "de"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from LAMBADA Multilingual dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []
        all_targets: list[str] = []

        try:
            # Try the multilingual config
            docs = self.load_dataset(
                dataset_name="EleutherAI/lambada_openai",
                dataset_config=self.language,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from LAMBADA ({self.language})")
        except Exception as e:
            log.error(f"Failed to load LAMBADA multilingual: {e}")
            return []

        # First pass: collect all targets
        for doc in docs:
            text = doc.get("text", "")
            if text:
                words = text.split()
                if words:
                    all_targets.append(words[-1])

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, all_targets)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], all_targets: list[str]
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            text = doc.get("text", "").strip()

            if not text:
                return None

            words = text.split()
            if len(words) < 2:
                return None

            # The task is to predict the last word
            context = " ".join(words[:-1])
            target_word = words[-1]

            task_prompt = f"""{context}"""

            # Get an incorrect word
            negative_candidates = [t for t in all_targets if t != target_word]
            if negative_candidates:
                incorrect = random.choice(negative_candidates)
            else:
                incorrect = "unknown"

            metadata = {
                "label": "lambada_multilingual",
                "source": "EleutherAI/lambada_openai",
                "language": self.language,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=target_word,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting LAMBADA pair: {exc}", exc_info=True)
            return None
