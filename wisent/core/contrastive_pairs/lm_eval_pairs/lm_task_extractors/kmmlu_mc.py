from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["KMMLUMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

# This extractor handles all kmmlu MC subtasks dynamically
task_names = ()  # Intentionally empty - will match any kmmlu MC task
class KMMLUMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for KMMLU multiple-choice benchmarks."""


    evaluator_name = "log_likelihoods"
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
            log.warning("No valid KMMLU MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            
            # Try different answer field names
            answer = doc.get("answer")
            if answer is None:
                answer = doc.get("answerKey")
            if answer is None:
                answer = doc.get("label")

            if not question or not choices or answer is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Parse answer index
            if isinstance(answer, int):
                answer_idx = answer
            elif isinstance(answer, str):
                if answer.upper() in ["A", "B", "C", "D", "E"]:
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    try:
                        answer_idx = int(answer)
                    except ValueError:
                        log.debug("Invalid answer format", extra={"doc": doc})
                        return None
            else:
                log.debug("Invalid answer type", extra={"doc": doc})
                return None

            if answer_idx < 0 or answer_idx >= len(choices):
                log.debug("Answer index out of range", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            formatted_question = f"Question: {question}\nAnswer:"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="kmmlu_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
