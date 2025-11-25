from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["AfrimgsmDirectAmhExtractor"]
_LOG = setup_logger(__name__)

task_names = ("afrimgsm_direct_amh",)

evaluator_name = "log_likelihoods"


class AfrimgsmDirectAmhExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Afrimgsm direct Amharic - math word problems with numeric answers."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="afrimgsm_direct_amh")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        from datasets import load_dataset
        try:
            dataset = load_dataset("masakhane/afrimgsm", "amh", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load afrimgsm dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "afrimgsm_direct_amh"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", "").strip()
            answer_number = doc.get("answer_number")            

            if not question or answer_number is None:
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            # Convert answer to string
            correct = str(answer_number)

            # Generate an incorrect answer
            try:
                num_val = float(answer_number)
                if num_val == 0:
                    incorrect = "1"
                elif num_val > 0:
                    incorrect = str(int(num_val + 1))
                else:
                    incorrect = str(int(num_val - 1))
            except (ValueError, TypeError):
                incorrect = "0"

            metadata = {"label": "afrimgsm_direct_amh"}

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
