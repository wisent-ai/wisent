from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ArcGenerationExtractor"]
_LOG = setup_logger(__name__)

task_names = ("arc_challenge_chat",)
class ArcGenerationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Arc generation benchmark (arc_challenge_chat)."""


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
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Arc generation pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", "").strip()
            choices = doc.get("choices", {})
            answer_key = doc.get("answerKey", "")

            if not question or not choices or not answer_key:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            choice_labels = choices.get("label", [])
            choice_texts = choices.get("text", [])

            if not choice_labels or not choice_texts:
                log.debug("Skipping doc due to missing choice data", extra={"doc": doc})
                return None

            # Find correct answer
            try:
                correct_idx = choice_labels.index(answer_key)
                correct_answer = choice_texts[correct_idx]
            except (ValueError, IndexError) as e:
                log.debug("Invalid answer key", extra={"error": str(e), "doc": doc})
                return None

            # Pick a wrong answer (first one that isn't correct)
            incorrect_answers = [text for i, text in enumerate(choice_texts) if i != correct_idx]
            if not incorrect_answers:
                log.debug("No incorrect answers available", extra={"doc": doc})
                return None

            incorrect_answer = incorrect_answers[0]

            # For generation task, format as conversational prompt
            formatted_question = f"Question: {question}\nAnswer:"

            positive_response = PositiveResponse(model_response=correct_answer)
            negative_response = NegativeResponse(model_response=incorrect_answer)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="arc_gen",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
