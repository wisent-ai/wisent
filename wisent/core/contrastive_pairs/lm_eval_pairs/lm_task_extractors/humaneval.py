from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["HumanEvalExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "humaneval",
    "humaneval_64",
    "humaneval_plus",
)

class HumanEvalExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the HumanEval benchmark - code generation tasks."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HumanEval docs.

        HumanEval schema:
            - prompt: str (function signature and docstring)
            - entry_point: str (function name)
            - canonical_solution: str (correct implementation)
            - test: str (test cases)

        Args:
            lm_eval_task_data: lm-eval task instance for HumanEval.
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
            log.warning("No valid HumanEval pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single HumanEval doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("task_id", "unknown"))

        try:
            prompt = str(doc.get("prompt", "")).strip()
            canonical_solution = str(doc.get("canonical_solution", "")).strip()
            entry_point = str(doc.get("entry_point", "")).strip()

            if not prompt or not canonical_solution or not entry_point:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Correct solution is the canonical implementation
            correct = canonical_solution

            # Incorrect solution: return a placeholder or incorrect implementation
            incorrect = "    pass  # Incomplete implementation"

            formatted_question = f"Complete the following Python function:\n\n{prompt}"

            metadata = {
                "label": "humaneval",
            }

            return self._build_pair(
                question=formatted_question,
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
