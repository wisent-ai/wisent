from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["KMMLUCotExtractor"]
_LOG = setup_logger(__name__)

# This extractor handles all kmmlu CoT subtasks dynamically
task_names = ()  # Intentionally empty - will match any kmmlu CoT task
class KMMLUCotExtractor(LMEvalBenchmarkExtractor):
    """Extractor for KMMLU CoT (chain-of-thought) benchmarks."""


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
            log.warning("No valid KMMLU CoT pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", "").strip()
            
            # Try different answer/target field names
            answer = doc.get("answer") or doc.get("target") or doc.get("answerKey")

            if not question or not answer:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            correct = str(answer).strip()

            # Generate incorrect answer (different letter or corrupted version)
            if correct in ["A", "B", "C", "D", "E"]:
                options = ["A", "B", "C", "D", "E"]
                options.remove(correct)
                incorrect = options[0]
            else:
                incorrect = "wrong answer"

            formatted_question = f"Question: {question}\nAnswer:"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="kmmlu_cot",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
