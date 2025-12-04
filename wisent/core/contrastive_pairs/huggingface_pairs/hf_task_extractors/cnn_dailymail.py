from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["CnnDailymailExtractor"]
_LOG = setup_logger(__name__)

task_names = ("cnn_dailymail",)

class CnnDailymailExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for CNN/DailyMail - summarization task."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="cnn_dailymail")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        from datasets import load_dataset
        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load cnn_dailymail dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "cnn_dailymail"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            article = doc.get("article", "").strip()
            highlights = doc.get("highlights", "").strip()

            if not article or not highlights:
                log.debug("Skipping doc due to missing article or highlights", extra={"doc": doc})
                return None

            correct_summary = highlights

            # Create synthetic negative by shuffling sentences
            import random
            sentences = [s.strip() for s in highlights.split('.') if s.strip()]
            if len(sentences) > 1:
                shuffled_sentences = sentences.copy()
                random.shuffle(shuffled_sentences)

                if shuffled_sentences == sentences:
                    shuffled_sentences = list(reversed(sentences))

                incorrect_summary = '. '.join(shuffled_sentences) + '.'
            else:
                incorrect_summary = "This is an incorrect summary."

            question = f"Summarize the following article:\n\n{article}"
            metadata = {"label": "cnn_dailymail"}

            return self._build_pair(
                question=question,
                correct=correct_summary,
                incorrect=incorrect_summary,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

