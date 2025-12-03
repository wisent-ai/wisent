from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["RealtoxicitypromptsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("realtoxicityprompts",)

class RealtoxicitypromptsExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Realtoxicityprompts benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Realtoxicityprompts docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Realtoxicityprompts.
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
            log.warning("No valid Realtoxicityprompts pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Realtoxicityprompts doc into a ContrastivePair.

        Schema: {prompt: {text: str, toxicity: float}, continuation: {text: str, toxicity: float}}
        Task: Generate non-toxic continuations for prompts
        Positive: Low toxicity continuation
        Negative: High toxicity continuation
        """
        log = bind(_LOG, doc_id=doc.get("filename", "unknown"))

        try:
            # Extract prompt text
            prompt_data = doc.get("prompt")
            if not prompt_data or not isinstance(prompt_data, dict):
                return None

            prompt_text = prompt_data.get("text", "").strip()
            if not prompt_text:
                return None

            # Extract continuation as positive response (should be low toxicity)
            continuation_data = doc.get("continuation")
            if not continuation_data or not isinstance(continuation_data, dict):
                return None

            continuation_text = continuation_data.get("text", "").strip()
            continuation_toxicity = continuation_data.get("toxicity", 0.0)

            if not continuation_text:
                return None

            # Create a toxic alternative as negative response
            # Since we don't have toxic examples, use a placeholder that indicates toxicity
            toxic_continuation = "[TOXIC CONTENT]"

            metadata = {
                "label": "realtoxicityprompts",
                "continuation_toxicity": continuation_toxicity,
            }

            return self._build_pair(
                question=prompt_text,
                correct=continuation_text,
                incorrect=toxic_continuation,
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
