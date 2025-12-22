from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Hrm8kExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "hrm8k",
    "hrm8k_en",
    "hrm8k_gsm8k",
    "hrm8k_gsm8k_en",
    "hrm8k_ksm",
    "hrm8k_ksm_en",
    "hrm8k_math",
    "hrm8k_math_en",
    "hrm8k_mmmlu",
    "hrm8k_mmmlu_en",
    "hrm8k_omni_math",
    "hrm8k_omni_math_en",
)
class Hrm8kExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Hrm8K benchmark."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Hrm8K docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Hrm8K.
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
            log.warning("No valid Hrm8K pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Hrm8K doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Hrm8K schema: question (str), answer (numeric), original (str), level, type
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Hrm8K format: question + numeric answer (math problems)
            if "question" in doc and "answer" in doc and not "choices" in doc:
                question = str(doc.get("question", "")).strip()
                answer = doc.get("answer")

                if not question or answer is None:
                    return None

                # Convert answer to string
                if isinstance(answer, (int, float)):
                    correct_answer = str(answer)
                else:
                    correct_answer = str(answer).strip()

                if not correct_answer:
                    return None

                # For numeric answers, create an incorrect answer by modifying the value
                try:
                    numeric_value = float(correct_answer)
                    # Create incorrect answer by adding 1
                    incorrect_answer = str(numeric_value + 1)
                except (ValueError, TypeError):
                    # If not numeric, use generic incorrect answer
                    incorrect_answer = "incorrect answer"

                metadata = {"label": "hrm8k"}
                return self._build_pair(
                    question=f"Question: {question}\nAnswer:",
                    correct=correct_answer,
                    incorrect=incorrect_answer,
                    metadata=metadata,
                )

            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("answer", doc.get("answerKey", ""))
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    answer_idx = int(answer) if answer else 0

            # Format 2: instruction + option_a/b/c/d + answer (MMMLU style)
            elif "instruction" in doc and "option_a" in doc:
                question = str(doc.get("instruction", "")).strip()
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')

            # Format 3: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "hrm8k"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "hrm8k",
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
