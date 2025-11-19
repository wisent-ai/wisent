from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MathQAExtractor"]
_LOG = setup_logger(__name__)


class MathQAExtractor(LMEvalBenchmarkExtractor):
    """Extractor for MathQA benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MathQA docs.

        Args:
            lm_eval_task_data: lm-eval task instance.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            # mathqa format: Problem, options, correct
            problem = doc.get("Problem", "").strip()
            options_str = doc.get("options", "")
            correct_key = doc.get("correct", "").strip().lower()

            if not problem or not options_str or not correct_key:
                _LOG.debug("Skipping: missing problem, options, or correct answer")
                return None

            # Parse options string like "a ) 38 , b ) 27.675 , c ) 30 , d ) data inadequate , e ) none of these"
            options_dict = {}
            parts = options_str.split(',')
            for part in parts:
                part = part.strip()
                if ')' in part:
                    key, value = part.split(')', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    options_dict[key] = value

            if correct_key not in options_dict:
                _LOG.debug(f"Skipping: correct key '{correct_key}' not in options")
                return None

            correct_answer = options_dict[correct_key]

            # Get an incorrect answer (any other option)
            incorrect_key = None
            for key in options_dict:
                if key != correct_key:
                    incorrect_key = key
                    break

            if incorrect_key is None:
                _LOG.debug("Skipping: no incorrect option available")
                return None

            incorrect_answer = options_dict[incorrect_key]

            # Format question
            formatted_question = f"Question: {problem}\nAnswer:"

            metadata = {
                "label": "mathqa",
                "source": getattr(task_data, "NAME", "mathqa"),
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            _LOG.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
