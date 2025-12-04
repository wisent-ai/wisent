from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FinancialTweetsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("financial_tweets",)

class FinancialTweetsExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Financial Tweets benchmark - tweet topic classification task.

    This is a Unitxt task where models classify financial tweets into topic categories.
    Format: source (classification prompt) + target (topic category)
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Financial Tweets docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Financial Tweets.
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

        # Group docs by category
        docs_by_category: dict[str, list[tuple[str, str]]] = {}

        for doc in docs:
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip().lower()

            if not source or not target:
                continue

            if target not in docs_by_category:
                docs_by_category[target] = []
            docs_by_category[target].append((source, target))

        # Create pairs: for each category, pair with a different category
        categories = list(docs_by_category.keys())

        for i, category in enumerate(categories):
            # Get incorrect category (next one in rotation)
            incorrect_category = categories[(i + 1) % len(categories)]

            # Pair docs from this category with incorrect category
            for source, correct_target in docs_by_category[category]:
                # Find an example from the incorrect category to use as wrong answer
                if docs_by_category[incorrect_category]:
                    _, incorrect_target = docs_by_category[incorrect_category][0]

                    metadata = {"label": "financial_tweets"}
                    pair = self._build_pair(
                        question=source,
                        correct=correct_target,
                        incorrect=incorrect_target,
                        metadata=metadata,
                    )
                    pairs.append(pair)

                    if max_items is not None and len(pairs) >= max_items:
                        break

            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Financial Tweets pairs extracted", extra={"task": task_name})

        return pairs

