from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["ClaimStanceTopicExtractor"]
_LOG = setup_logger(__name__)

task_names = ("claim_stance_topic",)

evaluator_name = "exact_match"


class ClaimStanceTopicExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Claim Stance Topic - stance detection."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="claim_stance_topic")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        from datasets import load_dataset
        try:
            dataset = load_dataset("ibm/claim_stance_dataset_v1", split="train")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load claim_stance_topic dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "claim_stance_topic"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            claim = doc.get("claim", doc.get("text", "")).strip()
            topic = doc.get("topic", "").strip()
            stance = doc.get("stance", doc.get("label", "")).strip()

            if not claim or not topic or not stance:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            correct = stance

            # Common stances: pro, con, neutral
            stances = ["pro", "con", "neutral"]
            incorrect_stances = [s for s in stances if s.lower() != correct.lower()]
            incorrect = incorrect_stances[0] if incorrect_stances else "neutral"

            question = f"Topic: {topic}\nClaim: {claim}\nWhat is the stance of this claim regarding the topic?"
            metadata = {"label": "claim_stance_topic"}

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
