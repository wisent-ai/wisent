from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ColaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("cola",)

class ColaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Cola benchmark - grammatical acceptability task."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        # CoLA format: sentence + label (0=unacceptable, 1=acceptable)
        for doc in docs:
            sentence = doc.get("sentence", "").strip()
            label = doc.get("label")

            if not sentence or label is None:
                continue

            # Create a prompt asking if the sentence is grammatically acceptable
            prompt = f"Sentence: {sentence}\n\nIs this sentence grammatically acceptable in English?\nAnswer:"

            # Label 1 = acceptable, Label 0 = unacceptable
            if label == 1:
                correct = "acceptable"
                incorrect = "unacceptable"
            else:
                correct = "unacceptable"
                incorrect = "acceptable"

            metadata = {"label": "cola"}

            pair = self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )
            pairs.append(pair)

            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

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
