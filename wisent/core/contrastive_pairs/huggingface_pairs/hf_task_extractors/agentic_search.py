from __future__ import annotations

from typing import Any
from wisent.core.cli.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

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
    evaluator_name = "search_augmented_qa"

    def __init__(self, flavor: str = "seal_0"):
        """
        Initialize SealQA extractor.

        Args:
            flavor: Benchmark flavor (seal_0, seal_hard, longseal)
        """
        super().__init__()
        self.flavor = flavor

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
                context_parts.append(f"Sources: {', '.join(urls[:3])}")
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


class FinSearchCompExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for FinSearchComp - Financial Search and Reasoning benchmark.

    Dataset: ByteSeedXpert/FinSearchComp on HuggingFace

    FinSearchComp is the first fully open-source agent benchmark for realistic,
    open-domain financial search and reasoning with 635 questions spanning
    global and Greater China markets.

    For financial search evaluation:
    - Positive (correct) = Accurate, up-to-date financial information
    - Negative (incorrect) = Outdated or incorrect financial data
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "financial_search"

    def __init__(self, region: str | None = None):
        """
        Initialize FinSearchComp extractor.

        Args:
            region: Optional filter (global, china)
        """
        super().__init__()
        self.region = region

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from FinSearchComp dataset.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # Load more docs than needed since some have null response_reference
            # Dataset has ~20% null values, so load 50% extra to be safe
            load_limit = int(max_items * 1.5) if max_items else None
            docs = self.load_dataset(
                dataset_name="ByteSeedXpert/FinSearchComp",
                split="test",
                limit=load_limit,
            )
            log.info(f"Loaded {len(docs)} examples from FinSearchComp")
        except Exception as e:
            log.warning(f"Failed to load FinSearchComp test split: {e}")
            # Try train split
            try:
                load_limit = int(max_items * 1.5) if max_items else None
                docs = self.load_dataset(
                    dataset_name="ByteSeedXpert/FinSearchComp",
                    split="train",
                    limit=load_limit,
                )
                log.info(f"Loaded {len(docs)} examples from FinSearchComp (train)")
            except Exception as e2:
                # Try validation split
                try:
                    load_limit = int(max_items * 1.5) if max_items else None
                    docs = self.load_dataset(
                        dataset_name="ByteSeedXpert/FinSearchComp",
                        split="validation",
                        limit=load_limit,
                    )
                    log.info(f"Loaded {len(docs)} examples from FinSearchComp (validation)")
                except Exception as e3:
                    log.error(f"Failed to load FinSearchComp: {e3}")
                    return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid FinSearchComp pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.

        FinSearchComp schema:
        - prompt: str (the financial query)
        - response_reference: str (ground truth answer)
        - label: str (category like Simple_Historical_Lookup)
        """
        try:
            question = (doc.get("prompt") or "").strip()
            correct_answer = (doc.get("response_reference") or "").strip()
            category = doc.get("label", "general")
            region = "global"

            if not question or not correct_answer:
                return None

            # Filter by region if specified
            if self.region and self.region.lower() != region.lower():
                return None

            task_prompt = f"""Financial Research Task ({category}):

{question}

Search for accurate, up-to-date financial information and provide a detailed answer.
Include sources and data timestamps where relevant."""

            # Create incorrect answer
            incorrect_answer = "I was unable to find current financial data for this query."

            metadata = {
                "label": "finsearchcomp",
                "source": "ByteSeedXpert/FinSearchComp",
                "category": category,
                "region": region,
                "is_financial_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting FinSearchComp pair: {exc}", exc_info=True)
            return None

