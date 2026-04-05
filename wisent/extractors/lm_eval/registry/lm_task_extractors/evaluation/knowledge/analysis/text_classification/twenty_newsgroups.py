from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TwentyNewsgroupsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("20_newsgroups",)

class TwentyNewsgroupsExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Twenty Newsgroups benchmark - text classification task."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)
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
            # Extract source (the classification prompt)
            source = doc.get("source", "").strip()
            if not source:
                log.debug("Skipping doc due to missing source", extra={"doc": doc})
                return None

            # Extract correct answer
            target = doc.get("target", "").strip()
            if not target:
                log.debug("Skipping doc due to missing target", extra={"doc": doc})
                return None

            # Get all possible categories from task_data JSON (preferred: clean list)
            # or fall back to parsing the source prompt
            categories = self._extract_categories_from_task_data(doc.get("task_data", ""))
            if not categories:
                categories = self._extract_categories_from_source(source)

            if not categories or target not in categories:
                log.debug("Could not extract categories or target not in categories", extra={"doc": doc})
                return None

            # Select an incorrect answer (any category that's not the target)
            incorrect = next((cat for cat in categories if cat != target), None)
            if not incorrect:
                log.debug("Could not find an incorrect category", extra={"doc": doc})
                return None

            metadata = {"label": "twenty_newsgroups"}

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
    def _extract_categories_from_task_data(task_data_str: str) -> list[str]:
        """Extract categories from task_data JSON string (clean list without trailing periods)."""
        if not task_data_str:
            return []
        try:
            data = json.loads(task_data_str)
            classes = data.get("classes", [])
            return [c for c in classes if c]
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _extract_categories_from_source(source: str) -> list[str]:
        """Extract the list of categories from the classification prompt."""
        # The source typically looks like:
        # "Classify the Topic of the following Text to one of these options: category1, category2, ..."
        if "options:" in source.lower():
            options_part = source.split("options:")[-1].split("\n")[0]
            categories = [cat.strip().rstrip(".") for cat in options_part.split(",")]
            return [cat for cat in categories if cat]
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
