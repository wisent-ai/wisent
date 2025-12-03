from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["SglueRteExtractor"]
_LOG = setup_logger(__name__)

task_names = ("sglue_rte",)

class SglueRteExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for SGLUE RTE - textual entailment task."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="sglue_rte")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("super_glue", "rte", split="validation")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load sglue_rte dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "sglue_rte"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"))

        try:
            premise = doc.get("premise", "").strip()
            hypothesis = doc.get("hypothesis", "").strip()
            label = doc.get("label")

            if not premise or not hypothesis or label is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # RTE labels: 0=entailment, 1=not_entailment
            if label == 0:
                correct = "entailment"
                incorrect = "not_entailment"
            else:
                correct = "not_entailment"
                incorrect = "entailment"

            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\n\nDoes the premise entail the hypothesis?"
            metadata = {"label": "sglue_rte"}

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
