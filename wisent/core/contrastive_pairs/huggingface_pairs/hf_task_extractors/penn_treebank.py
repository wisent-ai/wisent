from __future__ import annotations

import random
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["PennTreebankExtractor"]
_LOG = setup_logger(__name__)

task_names = ("penn_treebank", "ptb")

class PennTreebankExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Penn Treebank - language modeling perplexity task."""


    evaluator_name = "perplexity"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="penn_treebank")
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset("FALcon6/ptb_text_only", split="test", limit=max_items)

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "penn_treebank"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = doc.get("sentence", doc.get("page", doc.get("text", ""))).strip()
            words = text.split()

            if not text or len(words) < 3:
                log.debug("Skipping doc due to missing or short text", extra={"doc": doc})
                return None

            correct = text

            # Incorrect: shuffle word order
            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            incorrect = ' '.join(shuffled_words)

            # Skip if shuffle didn't change anything
            if incorrect == correct:
                return None

            prompt = "Which text is more coherent and grammatically correct?"
            metadata = {"label": "penn_treebank"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

