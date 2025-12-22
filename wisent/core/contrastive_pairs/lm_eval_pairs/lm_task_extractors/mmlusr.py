from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MmlusrExtractor"]
_LOG = setup_logger(__name__)

task_names = ("mmlusr", "mmlusr_question_and_answer", "mmlusr_question_only", "mmlusr_answer_only",
              "mmlusr_qa_stem", "mmlusr_qa_other", "mmlusr_qa_social_sciences", "mmlusr_qa_humanities")

class MmlusrExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Mmlusr benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mmlusr docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mmlusr.
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
            log.warning("No valid Mmlusr pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Mmlusr doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        MMLU-SR format (after process_docs):
        - questions: question text (note: typo in lm-eval, it's "questions" not "question")
        - choices: list of 4 choices
        - answer: letter "A", "B", "C", or "D"

        Raw format (before process_docs):
        - question: question text
        - choice1, choice2, choice3, choice4: the four choices
        - answer: numeric index (0-3)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = None
            choices = None
            answer_idx = None

            # Format 1: Raw HuggingFace format (column_0 through column_5)
            # column_0: question, column_1-4: choices, column_5: answer letter
            if "column_0" in doc and "column_1" in doc and "column_5" in doc:
                question = str(doc.get("column_0", "")).strip()
                choices = [
                    str(doc.get("column_1", "")).strip(),
                    str(doc.get("column_2", "")).strip(),
                    str(doc.get("column_3", "")).strip(),
                    str(doc.get("column_4", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("column_5", "A")
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    log.debug("Could not parse answer from column_5", extra={"answer": answer, "doc": doc})
                    return None

            # Format 2: Processed format (questions + choices array + answer letter)
            # Note: lm-eval has a typo where it outputs "questions" instead of "question"
            elif ("questions" in doc or "question" in doc) and "choices" in doc and "answer" in doc:
                question = str(doc.get("questions", doc.get("question", ""))).strip()
                choices = doc.get("choices", [])
                if not isinstance(choices, list):
                    log.debug("Choices is not a list", extra={"doc": doc})
                    return None

                answer = doc.get("answer", "")
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    # Answer is a letter like "A", "B", "C", "D"
                    answer_idx = ord(answer.upper()) - ord('A')
                elif isinstance(answer, int):
                    # Answer is already an index
                    answer_idx = answer
                else:
                    try:
                        answer_idx = int(answer)
                    except (ValueError, TypeError):
                        log.debug("Could not parse answer", extra={"answer": answer, "doc": doc})
                        return None

            # Format 3: Named format (question + choice1/2/3/4 + answer)
            elif "question" in doc and "choice1" in doc:
                question = str(doc.get("question", "")).strip()
                choices = [
                    str(doc.get("choice1", "")).strip(),
                    str(doc.get("choice2", "")).strip(),
                    str(doc.get("choice3", "")).strip(),
                    str(doc.get("choice4", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", 0)
                if isinstance(answer, int):
                    answer_idx = answer
                else:
                    try:
                        answer_idx = int(answer)
                    except (ValueError, TypeError):
                        log.debug("Could not parse answer", extra={"answer": answer, "doc": doc})
                        return None
            else:
                log.debug("Skipping doc without required fields", extra={"doc": doc})
                return None

            if not question or not choices or len(choices) < 2 or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"question": question, "num_choices": len(choices) if choices else 0, "answer_idx": answer_idx, "doc": doc},
                )
                return None

            # Build prompt - raw question without MC formatting
            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            prompt = question

            metadata = {
                "label": "mmlusr",
            }

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
