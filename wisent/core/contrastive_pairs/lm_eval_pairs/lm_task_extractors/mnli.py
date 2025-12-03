from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MnliExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "mnli",
    "mnli_mismatch",
)

class MnliExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Mnli benchmark."""


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
        """
        Convert a single MNLI doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        MNLI format (natural language inference):
        - premise: the premise statement
        - hypothesis: the hypothesis statement
        - label: 0 (entailment/True), 1 (neutral/Neither), or 2 (contradiction/False)
        """
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"))

        try:
            # MNLI format
            premise = str(doc.get("premise", "")).strip()
            hypothesis = str(doc.get("hypothesis", "")).strip()
            label = doc.get("label", None)

            if not premise or not hypothesis or label is None:
                log.debug("Skipping doc due to missing premise/hypothesis/label", extra={"doc": doc})
                return None

            # MNLI labels: 0 = entailment (True), 1 = neutral (Neither), 2 = contradiction (False)
            choices = ["True", "Neither", "False"]

            if not isinstance(label, int) or not (0 <= label < len(choices)):
                log.debug("Invalid label", extra={"label": label, "doc": doc})
                return None

            # Add period to hypothesis if not present
            hypothesis_formatted = hypothesis + ("" if hypothesis.endswith(".") else ".")

            # Format exactly as lm-eval does it
            prompt = f"{premise}\nQuestion: {hypothesis_formatted} True, False or Neither?\nAnswer:"

            correct = choices[label]
            incorrect_idx = (label + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {"label": "mnli"}

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
