from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SquadCompletionExtractor"]
_LOG = setup_logger(__name__)

task_names = ("squad_completion",)

class SquadCompletionExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Squad Completion benchmark."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Squad Completion docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Squad Completion.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)

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
            log.warning("No valid Squad Completion pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Squad Completion doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("doc_id", doc.get("id", "unknown")))

        try:
            # Squad Completion format: text (context ending incomplete) + value (correct completion)
            # e.g., {"text": "...The NFL team that represented the AFC at Super Bowl 50 was the", "value": "Denver Broncos"}

            if "text" not in doc or "value" not in doc:
                log.debug(
                    "Skipping doc due to missing 'text' or 'value' fields",
                    extra={"doc": doc},
                )
                return None

            text = str(doc.get("text", "")).strip()
            correct_value = str(doc.get("value", "")).strip()

            if not text or not correct_value:
                log.debug(
                    "Skipping doc due to empty text or value",
                    extra={"doc": doc},
                )
                return None

            # For squad_completion, the prompt is the incomplete sentence
            # Positive: correct completion
            # Negative: use a generic incorrect completion
            prompt = text

            metadata = {
                "label": "squad_completion",
                "title": doc.get("title", "unknown"),
            }

            return self._build_pair(
                question=prompt,
                correct=correct_value,
                incorrect="Unknown",
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
