from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ANLIExtractor"]
_LOG = setup_logger(__name__)

task_names = ("anli", "anli_r1", "anli_r2", "anli_r3")

class ANLIExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Adversarial NLI (ANLI) benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from ANLI docs.

        ANLI schema:
            - premise: str (the premise)
            - hypothesis: str (the hypothesis)
            - label: int (0=entailment, 1=neutral, 2=contradiction)

        Args:
            lm_eval_task_data: lm-eval task instance for ANLI.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid ANLI pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            premise = doc.get("premise", "").strip()
            hypothesis = doc.get("hypothesis", "").strip()
            label = doc.get("label", -1)

            if not all([premise, hypothesis]) or label == -1:
                _LOG.debug("Skipping: missing premise, hypothesis, or label")
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"{premise}\nQuestion: {hypothesis} True, False, or Neither?"

            # Map label to answer
            label_map = {0: "True", 1: "Neither", 2: "False"}
            correct_answer = label_map.get(label)

            if correct_answer is None:
                _LOG.debug(f"Unknown label: {label}")
                return None

            # Get an incorrect answer (pick first one that's different)
            incorrect_answer = None
            for label_key, answer in label_map.items():
                if label_key != label:
                    incorrect_answer = answer
                    break

            if not incorrect_answer:
                _LOG.debug("Could not generate incorrect answer")
                return None

            metadata = {
                "label": "anli",
                "source": getattr(task_data, "NAME", "anli"),
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            _LOG.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
