from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ChartqaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("chartqa", "chartqa_llama", "chartqa_llama_90")

class ChartqaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Chartqa benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Chartqa docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Chartqa.
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
            log.warning("No valid Chartqa pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Chartqa doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
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

            # Format 3: ChartQA format (query + label list)
            elif "query" in doc and "label" in doc:
                question = str(doc.get("query", "")).strip()
                label = doc.get("label", [])

                # label is a list of acceptable answers, use the first one
                if not isinstance(label, list) or not label:
                    log.debug("Skipping doc - label is not a valid list", extra={"doc": doc})
                    return None

                correct_answer = str(label[0]).strip()
                if not correct_answer:
                    log.debug("Skipping doc - empty correct answer", extra={"doc": doc})
                    return None

                # Create synthetic negative based on answer type
                incorrect_answer = self._create_synthetic_negative(correct_answer)

                metadata = {"label": "chartqa"}
                return self._build_pair(
                    question=f"Question: {question}",
                    correct=correct_answer,
                    incorrect=incorrect_answer,
                    metadata=metadata,
                )

            # Format 4: query/prompt + answer (fallback for other tasks)
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    incorrect_answer = self._create_synthetic_negative(correct_answer)
                    metadata = {"label": "chartqa"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect=incorrect_answer,
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
                "label": "chartqa",
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
    def _create_synthetic_negative(correct_answer: str) -> str:
        """
        Create a synthetic negative answer based on the correct answer type.

        Args:
            correct_answer: The correct answer to base the negative on.

        Returns:
            A synthetic negative answer.
        """
        import random

        correct_answer = correct_answer.strip()

        # Handle Yes/No questions
        if correct_answer.lower() in ["yes", "no"]:
            return "No" if correct_answer.lower() == "yes" else "Yes"

        # Handle numeric answers (with or without % or decimal points)
        # Remove % sign if present for numeric detection
        numeric_part = correct_answer.rstrip('%').strip()

        # Check if it's a number (including decimals)
        try:
            num = float(numeric_part)

            # For percentages, modify the number
            if correct_answer.endswith('%'):
                # Add/subtract a random amount between 10-30%
                change = random.choice([-30, -20, -10, 10, 20, 30])
                new_num = max(0, min(100, num + change))  # Keep within 0-100 range
                return f"{new_num:.1f}%"

            # For integers
            if '.' not in numeric_part:
                num_int = int(num)
                # Modify by +/- 1 to 5
                change = random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                new_num = max(0, num_int + change)
                return str(new_num)

            # For decimals/ratios
            # Modify by +/- 0.05 to 0.2
            change = random.choice([-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2])
            new_num = max(0, num + change)
            return f"{new_num:.2f}"

        except ValueError:
            # Not a number, handle as text
            pass

        # For entity/text answers, use a generic incorrect answer
        # We can't create a meaningful incorrect entity without knowing the chart
        return "Incorrect Value"

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
