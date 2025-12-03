from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ModelWrittenEvalsExtractor"]
_LOG = setup_logger(__name__)

# Model Written Evals includes multiple sub-benchmarks
task_names = ("model_written_evals", "advanced_ai_risk", "persona", "sycophancy", "winogenerated")

class ModelWrittenEvalsExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Model Written Evals benchmark (advanced_ai_risk, persona, sycophancy, winogenerated)."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Model Written Evals docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Model Written Evals.
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
            log.warning("No valid Model Written Evals pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Model Written Evals doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Model Written Evals format (advanced_ai_risk, persona, sycophancy, etc.):
        - question: the question text
        - answer_matching_behavior: the correct answer (target is always 0)
        - answer_not_matching_behavior: the incorrect answer
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Model Written Evals format
            if "question" in doc and "answer_matching_behavior" in doc and "answer_not_matching_behavior" in doc:
                question = str(doc.get("question", "")).strip()
                correct = str(doc.get("answer_matching_behavior", "")).strip()
                incorrect = str(doc.get("answer_not_matching_behavior", "")).strip()

                if not question or not correct or not incorrect:
                    log.debug("Skipping doc with missing fields", extra={"doc": doc})
                    return None

                # Format prompt as lm-eval does: "Human: {question}\n\nAssistant:"
                prompt = f"Human: {question}\n\nAssistant:"

                metadata = {"label": "model_written_evals"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without required fields", extra={"doc": doc})
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
