from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["BrowseCompExtractor"]

log = setup_logger(__name__)


class BrowseCompExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for BrowseComp - web browsing/search agent benchmark by OpenAI.

    BrowseComp evaluates LLMs' ability to perform web browsing and search tasks.
    Tests include finding specific information, navigating websites, and
    synthesizing information from multiple sources.

    For web browsing evaluation:
    - Positive (correct) = Accurate information retrieval with correct navigation
    - Negative (incorrect) = Wrong information or failed navigation
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "web_browsing_accuracy"

    def __init__(self, language: str = "en"):
        """
        Initialize BrowseComp extractor.

        Args:
            language: Language code (en for English, zh for Chinese)
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from BrowseComp examples.

        Uses Tevatron/browsecomp-plus dataset from HuggingFace.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="Tevatron/browsecomp-plus",
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from browsecomp-plus")

            for doc in docs:
                pair = self._extract_pair_from_doc(doc)
                if pair is not None:
                    pairs.append(pair)
                    if max_items is not None and len(pairs) >= max_items:
                        break

        except Exception as e:
            log.error(f"Failed to load browsecomp-plus: {e}")
            return []

        if not pairs:
            log.warning("No valid BrowseComp pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.
        
        browsecomp-plus schema:
        - query_id: str
        - query: str (the question)
        - answer: str (ground truth answer)
        - evidence_docs: list (documents with evidence)
        - gold_docs: list (gold standard documents)
        - negative_docs: list (distractor documents)
        """
        try:
            query = doc.get("query", "").strip()
            correct_answer = doc.get("answer", "").strip()
            query_id = doc.get("query_id", "")

            if not query or not correct_answer:
                return None

            # Build the web browsing task prompt
            task_prompt = f"""Web Search Task: {query}

Please search the web and provide accurate, up-to-date information. Include:
- The source(s) of your information
- Relevant details and context
- Any caveats about data freshness"""

            # Create incorrect answer (opposite or unrelated)
            incorrect_answer = f"I could not find relevant information about this query."

            metadata = {
                "label": "browsecomp",
                "source": "Tevatron/browsecomp-plus",
                "query_id": query_id,
                "language": self.language,
                "is_web_browsing_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None





