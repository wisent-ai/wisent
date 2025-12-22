from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["StoryclozeExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "storycloze",
    "storycloze_2016",
    "storycloze_2018",
)

class StoryclozeExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Storycloze benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Storycloze docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Storycloze.
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
            log.warning("No valid Storycloze pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Storycloze doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: Storycloze native format (input_sentence_1-4 + sentence_quiz1/2 + answer_right_ending)
            if "input_sentence_1" in doc and "sentence_quiz1" in doc:
                # Build context from the four input sentences
                context_parts = []
                for i in range(1, 5):
                    sent = doc.get(f"input_sentence_{i}", "")
                    if sent:
                        context_parts.append(str(sent).strip())

                if not context_parts:
                    log.debug(
                        "Skipping doc due to missing input sentences",
                        extra={"doc": doc},
                    )
                    return None

                context = " ".join(context_parts)

                # Get the two possible endings
                choice1 = str(doc.get("sentence_quiz1", "")).strip()
                choice2 = str(doc.get("sentence_quiz2", "")).strip()

                if not choice1 or not choice2:
                    log.debug(
                        "Skipping doc due to missing sentence_quiz choices",
                        extra={"doc": doc},
                    )
                    return None

                # answer_right_ending is 1 or 2 (1-indexed)
                answer_right_ending = doc.get("answer_right_ending")
                if answer_right_ending is None:
                    log.debug(
                        "Skipping doc due to missing answer_right_ending",
                        extra={"doc": doc},
                    )
                    return None

                # Convert to 0-indexed
                answer_idx = int(answer_right_ending) - 1

                if answer_idx == 0:
                    correct = choice1
                    incorrect = choice2
                elif answer_idx == 1:
                    correct = choice2
                    incorrect = choice1
                else:
                    log.debug(
                        "Skipping doc due to invalid answer_right_ending",
                        extra={"doc": doc, "answer_right_ending": answer_right_ending},
                    )
                    return None

                metadata = {
                    "label": "storycloze",
                }

                return self._build_pair(
                    question=context,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 2: question + choices + answer
            elif "question" in doc and "choices" in doc:
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
                    "label": "storycloze",
                }

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 3: instruction + option_a/b/c/d + answer (MMMLU style)
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
                    "label": "storycloze",
                }

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 4: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "storycloze"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            else:
                log.debug(
                    "Skipping doc due to unrecognized format",
                    extra={"doc": doc},
                )
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
