from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ToxigenExtractor"]
_LOG = setup_logger(__name__)

task_names = ("toxigen",)

class ToxigenExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Toxigen benchmark.

    Toxigen is a binary classification task (Yes/No) for detecting hateful statements.
    """


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Toxigen docs using lm-eval's task methods.

        Args:
            lm_eval_task_data: lm-eval task instance for Toxigen.
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
            try:
                # Use lm_eval's task methods to format the question and get choices
                question = lm_eval_task_data.doc_to_text(doc)
                choices = lm_eval_task_data.doc_to_choice(doc)
                target_idx = lm_eval_task_data.doc_to_target(doc)

                if not question or not choices or target_idx is None:
                    continue

                if not (0 <= target_idx < len(choices)):
                    continue

                correct = choices[target_idx]
                incorrect_idx = (target_idx + 1) % len(choices)
                incorrect = choices[incorrect_idx]

                positive_response = PositiveResponse(model_response=correct)
                negative_response = NegativeResponse(model_response=incorrect)
                pair = ContrastivePair(
                    prompt=question,
                    positive_response=positive_response,
                    negative_response=negative_response,
                    label="toxigen",
                )
                pairs.append(pair)

                if max_items is not None and len(pairs) >= max_items:
                    break

            except Exception as exc:
                log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
                continue

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Toxigen pairs extracted", extra={"task": task_name})

        return pairs
