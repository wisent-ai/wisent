from __future__ import annotations

import random
from wisent.core.cli_logger import setup_logger
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor


__all__ = ["Wikitext103Extractor"]
log = setup_logger(__name__)

task_names = ("wikitext103",)


class Wikitext103Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for WikiText-103 - language modeling perplexity task."""

    evaluator_name = "perplexity"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="Salesforce/wikitext",
            dataset_config="wikitext-103-v1",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "wikitext103"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            text = doc.get("text", doc.get("page", "")).strip()

            if not text or len(text.split()) < 10:
                log.debug("Skipping doc due to missing or short text", extra={"doc": doc})
                return None

            # For perplexity tasks, create pairs by corrupting the text
            # Split text into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                log.debug("Not enough sentences to create pair", extra={"doc": doc})
                return None

            # Use all sentences except the last as context
            context = '. '.join(sentences[:-1]) + '.'

            # Correct: original text
            correct = text

            # Incorrect: shuffle word order in last sentence
            last_sentence = sentences[-1]
            words = last_sentence.split()
            if len(words) < 3:
                return None

            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            shuffled_sentence = ' '.join(shuffled_words)

            incorrect_sentences = sentences[:-1] + [shuffled_sentence]
            incorrect = '. '.join(incorrect_sentences) + '.'

            prompt = f"Complete the text coherently:\n\n{context}"
            metadata = {"label": "wikitext103"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

