from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CoqcatExtractor"]
_LOG = setup_logger(__name__)

task_names = ("coqcat",)

class CoqcatExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Coqcat benchmark - reading comprehension task."""


    evaluator_name = "generation"
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
            doc_pairs = self._extract_pairs_from_doc(doc)
            pairs.extend(doc_pairs)
            if max_items is not None and len(pairs) >= max_items:
                pairs = pairs[:max_items]
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pairs_from_doc(self, doc: dict[str, Any]) -> list[ContrastivePair]:
        """
        Extract multiple pairs from a single doc.
        Each doc contains a story and multiple question-answer pairs.
        """
        pairs = []

        try:
            story = doc.get("story", "").strip()
            questions = doc.get("questions", [])
            answers_dict = doc.get("answers", {})

            if not story or not questions or not answers_dict:
                return pairs

            # Get the list of correct answers
            correct_answers = answers_dict.get("input_text", [])

            if len(questions) != len(correct_answers):
                return pairs

            # Create a pair for each question
            for i, question in enumerate(questions):
                if i >= len(correct_answers):
                    break

                question_text = str(question).strip()
                correct_answer = str(correct_answers[i]).strip()

                if not question_text or not correct_answer:
                    continue

                # For the incorrect answer, use another answer from a different question in the same story
                # This ensures the incorrect answer is plausible but wrong for this specific question
                incorrect_idx = (i + 1) % len(correct_answers)
                incorrect_answer = str(correct_answers[incorrect_idx]).strip()

                # Format: Story + Question
                prompt = f"Story: {story}\n\nQuestion: {question_text}"

                metadata = {"label": "coqcat"}

                pair = self._build_pair(
                    question=prompt,
                    correct=correct_answer,
                    incorrect=incorrect_answer,
                    metadata=metadata,
                )
                pairs.append(pair)

        except Exception as exc:
            log = bind(_LOG, doc_id=doc.get("id", "unknown"))
            log.error("Error extracting pairs from doc", exc_info=exc, extra={"doc": doc})

        return pairs

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
