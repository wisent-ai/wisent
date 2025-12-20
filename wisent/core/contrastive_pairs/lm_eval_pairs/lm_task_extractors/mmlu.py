from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MMLUExtractor"]
_LOG = setup_logger(__name__)

task_names = ("mmlu",)
class MMLUExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the MMLU (Massive Multitask Language Understanding) and MMMLU (multilingual variant) benchmarks."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MMLU/MMMLU docs.

        MMLU schema:
            - question: str
            - choices: list[str] with 4 possible choices (A, B, C, D)
            - answer: int (0-3) indicating the correct answer index

        MMMLU schema (multilingual variant):
            - instruction: str (the question)
            - option_a, option_b, option_c, option_d: str (individual choices)
            - answer: str (letter A-D indicating correct answer)

        Args:
            lm_eval_task_data: lm-eval task instance for MMLU.
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
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid MMLU pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MMLU/MMMLU doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Check for MMMLU format first (uses instruction + option_a/b/c/d)
            if "instruction" in doc and "option_a" in doc:
                question = str(doc.get("instruction", "")).strip()
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                # Filter out empty choices
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                # MMMLU answer is always a letter
                answer_idx = ord(str(answer).upper()) - ord('A')
            else:
                # Standard MMLU format
                question = str(doc.get("question", "")).strip()
                choices = doc.get("choices", [])
                answer = doc.get("answer", None)

                # Handle different answer formats (could be int or str)
                if isinstance(answer, str):
                    # If answer is a letter like 'A', 'B', 'C', 'D'
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    # If answer is already an index
                    answer_idx = int(answer)

            if not question or not choices or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx]
            # Pick a different wrong answer
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "mmlu",
            }

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
