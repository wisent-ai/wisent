from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Anagrams2Extractor"]
_LOG = setup_logger(__name__)

task_names = ("anagrams2",)

class Anagrams2Extractor(LMEvalBenchmarkExtractor):
    """Extractor for Anagrams2 benchmark - anagram solving task."""


    evaluator_name = "exact_match"
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

        # For anagrams, we need to use other docs' completions as incorrect answers
        # First, extract all valid completions
        valid_docs = []
        for doc in docs:
            context = doc.get("context", "").strip()
            completion = doc.get("completion", "").strip()
            if context and completion:
                valid_docs.append((context, completion))

        if len(valid_docs) < 2:
            log.warning("Not enough valid docs to create pairs", extra={"valid_count": len(valid_docs)})
            return pairs

        # Create pairs using next doc's completion as incorrect answer
        for i, (context, correct_completion) in enumerate(valid_docs):
            # Use the next doc's completion as incorrect answer
            incorrect_completion = valid_docs[(i + 1) % len(valid_docs)][1]

            metadata = {"label": "anagrams2"}
            pair = self._build_pair(
                question=context,
                correct=correct_completion,
                incorrect=incorrect_completion,
                metadata=metadata,
            )
            pairs.append(pair)

            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

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
