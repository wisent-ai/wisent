from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FrenchBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "french_bench_arc_challenge",
    "french_bench_boolqa",
    "french_bench_hellaswag",
    "french_bench_multifquad",
    "french_bench_reading_comp",
    "french_bench_topic_based_nli",
    "french_bench_trivia",
    "french_bench_vocab",
    "french_bench_xnli"
)
class FrenchBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for French Bench multiple-choice benchmarks."""


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
            log.warning("No valid French Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            choices = doc.get("choices", [])
            answer_key = doc.get("answerKey", "")

            if not question or not choices or not answer_key:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Find correct answer index
            answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            answer_idx = answer_map.get(answer_key.upper())

            if answer_idx is None or answer_idx >= len(choices):
                log.debug("Invalid answer key", extra={"doc": doc, "answer": answer_key})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            formatted_question = f"Question: {question}\nRéponse:"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="french_bench_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
