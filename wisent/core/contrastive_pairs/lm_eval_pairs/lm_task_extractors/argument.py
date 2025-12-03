from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ArgumentExtractor"]
_LOG = setup_logger(__name__)

task_names = ("argument", "argument_topic")

class ArgumentExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Argument benchmark - text classification task."""


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

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Argument is a generation task with source/target format
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip()

            if not source or not target:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            # Extract available categories from the source prompt
            categories = self._extract_categories_from_source(source)
            if not categories:
                log.debug("Could not extract categories from source", extra={"source": source})
                return None

            # Verify target is in categories
            if target not in categories:
                log.debug("Target not found in categories", extra={"target": target, "categories": categories})
                return None

            # Select incorrect answer (any category that's not the target)
            incorrect = next((cat for cat in categories if cat != target), None)
            if not incorrect:
                log.debug("Could not find incorrect category", extra={"target": target, "categories": categories})
                return None

            metadata = {"label": "argument"}

            return self._build_pair(
                question=source,
                correct=target,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _extract_categories_from_source(source: str) -> list[str]:
        """
        Extract category options from the source prompt.

        Argument format: "Classify the Topic of the following Argument to one of these options: affirmative action, algorithmic trading, ..."
        """
        # Look for "options:" pattern (case insensitive search but preserve original case)
        if "options:" in source.lower():
            # Find the position and extract from original string (to preserve case)
            lower_source = source.lower()
            idx = lower_source.find("options:")
            if idx != -1:
                # Get text after "options:"
                options_text = source[idx + len("options:"):]
                # Split at the first period or newline to get just the category list
                # Format: " affirmative action, algorithmic trading, ...\nArgument:\n..."
                end_idx = len(options_text)
                for delimiter in [".\n", ".\r", ".  ", ". "]:
                    pos = options_text.find(delimiter)
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
                options_text = options_text[:end_idx].strip()

                # Remove trailing period if present
                if options_text.endswith("."):
                    options_text = options_text[:-1]

                # Split by comma and clean up
                categories = [cat.strip() for cat in options_text.split(",") if cat.strip()]
                return categories

        return []

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
