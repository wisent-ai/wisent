from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["AgNewsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("ag_news",)

evaluator_name = "exact_match"


class AgNewsExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for AG News - text classification task."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="ag_news")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        from datasets import load_dataset
        try:
            dataset = load_dataset("fancyzhx/ag_news", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load ag_news dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "ag_news"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = doc.get("text", "").strip()
            label = doc.get("label")

            if not text or label is None:
                log.debug("Skipping doc due to missing text or label", extra={"doc": doc})
                return None

            # AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
            label_map = {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Sci/Tech"
            }

            if label not in label_map:
                log.debug(f"Unknown label: {label}", extra={"doc": doc})
                return None

            correct = label_map[label]

            # Pick a different label as incorrect
            incorrect_labels = [l for l in label_map.values() if l != correct]
            incorrect = incorrect_labels[0] if incorrect_labels else "World"

            question = f"Classify the following news article into one of these categories: World, Sports, Business, Sci/Tech\n\n{text}"
            metadata = {"label": "ag_news"}

            return self._build_pair(
                question=question,
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
