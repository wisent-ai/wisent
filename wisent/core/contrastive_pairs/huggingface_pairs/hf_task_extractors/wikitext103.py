from __future__ import annotations

import random
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["Wikitext103Extractor"]
_LOG = setup_logger(__name__)

task_names = ("wikitext103",)

evaluator_name = "log_likelihoods"


class Wikitext103Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for WikiText-103 - language modeling perplexity task."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="wikitext103")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load wikitext103 dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(dataset)})

        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "wikitext103"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

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

            # Take first few sentences as context
            context = '. '.join(sentences[:2]) + '.'

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

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
