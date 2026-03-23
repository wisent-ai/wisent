from __future__ import annotations

from typing import Any, Optional
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.config_tools.constants import DISPLAY_TOP_N_TINY

__all__ = ["BrowseCompExtractor", "SealExtractor", "FinSearchCompExtractor"]

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
    evaluator_name = "browsecomp"

    def __init__(self, language: Optional[str] = None):
        """
        Initialize BrowseComp extractor.

        Args:
            language: Language code (en for English, zh for Chinese)
        """
        super().__init__()
        self.language = language if language is not None else "en"

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


class SealExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SealQA - Search-Augmented Language Model benchmark.

    Dataset: vtllms/sealqa on HuggingFace

    SealQA evaluates reasoning under noisy, conflicting, and ambiguous search
    results. Includes three flavors: Seal-0, Seal-Hard, and LongSeal.

    For search-augmented QA evaluation:
    - Positive (correct) = Accurate answer despite noisy search context
    - Negative (incorrect) = Wrong answer due to misleading context
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "seal"

    def __init__(self, flavor: Optional[str] = None):
        """
        Initialize SealQA extractor.

        Args:
            flavor: Benchmark flavor (seal_0, seal_hard, longseal)
        """
        super().__init__()
        self.flavor = flavor if flavor is not None else "seal_0"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SealQA dataset.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="vtllms/sealqa",
                dataset_config=self.flavor,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from SealQA")
        except Exception as e:
            log.warning(f"Failed to load SealQA test split: {e}")
            # Try train split
            try:
                docs = self.load_dataset(
                    dataset_name="vtllms/sealqa",
                    dataset_config=self.flavor,
                    split="train",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from SealQA (train)")
            except Exception as e2:
                # Try validation split
                try:
                    docs = self.load_dataset(
                        dataset_name="vtllms/sealqa",
                        dataset_config=self.flavor,
                        split="validation",
                        limit=max_items,
                    )
                    log.info(f"Loaded {len(docs)} examples from SealQA (validation)")
                except Exception as e3:
                    log.error(f"Failed to load SealQA: {e3}")
                    return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SealQA pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.

        SealQA schema:
        - question: str (the query)
        - answer: str (ground truth answer)
        - urls: list (source URLs)
        - search_results: str (type like "conflicting")
        - topic: str (topic category)
        """
        try:
            question = doc.get("question", "").strip()
            correct_answer = doc.get("answer", "").strip()
            urls = doc.get("urls", [])
            search_type = doc.get("search_results", "")
            topic = doc.get("topic", "")

            if not question or not correct_answer:
                return None

            # Build context from available info
            context_parts = []
            if urls:
                context_parts.append(f"Sources: {', '.join(urls[:DISPLAY_TOP_N_TINY])}")
            if search_type:
                context_parts.append(f"Search result type: {search_type}")
            context_text = "\n".join(context_parts)

            task_prompt = f"""Search-Augmented QA Task ({topic}):

Question: {question}

{context_text}

Provide the most accurate answer."""

            # Create incorrect answer
            incorrect_answer = "I cannot determine the answer from the provided search results."

            metadata = {
                "label": f"sealqa_{self.flavor}",
                "source": "vtllms/sealqa",
                "flavor": self.flavor,
                "is_search_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting SealQA pair: {exc}", exc_info=True)
            return None



# Re-export from split module
from wisent.extractors.hf.hf_task_extractors.agentic_search_finsearch import (
    FinSearchCompExtractor,
)
