from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MelaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("mela", "mela_en", "mela_zh", "mela_it", "mela_ru", "mela_de", "mela_fr", "mela_es", "mela_ja", "mela_ar")

class MelaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Mela benchmark."""

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # mela format: sentence + label (0=unacceptable, 1=acceptable)
            if "sentence" in doc and "label" in doc:
                sentence = str(doc.get("sentence", "")).strip()
                label = doc.get("label")

                if not sentence or label is None:
                    log.debug("Skipping doc with missing sentence or label", extra={"doc": doc})
                    return None

                # label 0 = B (Unacceptable), label 1 = A (Acceptable)
                choices = ["Acceptable", "Unacceptable"]
                # Map label to answer: label 0 -> B (index 1), label 1 -> A (index 0)
                answer_idx = 0 if label == 1 else 1

                correct = choices[answer_idx]
                incorrect_idx = 1 - answer_idx
                incorrect = choices[incorrect_idx]

                prompt = f"Sentence: {sentence}\nDetermine whether this sentence is acceptable or unacceptable?"
                metadata = {"label": "mela"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without sentence/label fields", extra={"doc": doc})
            return None

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
