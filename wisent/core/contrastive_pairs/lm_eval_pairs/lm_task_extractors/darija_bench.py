from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["DarijaBenchExtractor"]
_LOG = setup_logger(__name__)


class DarijaBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Darija Bench benchmark."""

    task_names = (
        "darija_bench",
        "darija_sentiment",
        "darija_sentiment_mac",
        "darija_sentiment_myc",
        "darija_sentiment_msac",
        "darija_sentiment_msda",
        "darija_sentiment_electrom",
        "darija_sentiment_tasks",
        "darija_summarization",
        "darija_summarization_task",
        "darija_translation",
        "darija_translation_doda",
        "darija_translation_flores",
        "darija_translation_madar",
        "darija_translation_seed",
        "darija_translation_tasks_doda",
        "darija_translation_tasks_flores",
        "darija_translation_tasks_madar",
        "darija_translation_tasks_seed",
        "darija_transliteration",
        "darija_transliteration_tasks",
    )
    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Darija Bench docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Darija Bench.
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
            log.warning("No valid Darija Bench pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Darija Bench doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 0: messages format (darija_sentiment)
            # messages[0]['content'] has question and choices, messages[1]['content'] has answer
            if "messages" in doc and isinstance(doc["messages"], list) and len(doc["messages"]) >= 2:
                user_msg = doc["messages"][0].get("content", "")
                assistant_msg = doc["messages"][1].get("content", "").strip()

                # Extract question and parse options from user message
                # Format: "شنو هو الإحساس ديال هاد الجملة؟\nالعبارة: ...\n الإحتمالات:\n-سلبي\n-ايجابي\n-ماكينش إحساس"
                if user_msg and assistant_msg:
                    # Extract the choices from the user message
                    lines = user_msg.split('\n')
                    choices = []
                    for line in lines:
                        if line.strip().startswith('-'):
                            choice = line.strip().lstrip('-').strip()
                            if choice:
                                choices.append(choice)

                    # Find which choice matches the answer
                    if choices and assistant_msg:
                        try:
                            answer_idx = choices.index(assistant_msg)
                            question = user_msg
                        except ValueError:
                            # Try partial match
                            for i, choice in enumerate(choices):
                                if choice in assistant_msg or assistant_msg in choice:
                                    answer_idx = i
                                    question = user_msg
                                    break

                    if question and choices and answer_idx is not None:
                        correct = choices[answer_idx]
                        incorrect_idx = (answer_idx + 1) % len(choices)
                        incorrect = choices[incorrect_idx]

                        metadata = {"label": "darija_bench"}
                        return self._build_pair(
                            question=question,
                            correct=correct,
                            incorrect=incorrect,
                            metadata=metadata,
                        )

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
                    metadata = {"label": "darija_bench"}
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
                "label": "darija_bench",
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
