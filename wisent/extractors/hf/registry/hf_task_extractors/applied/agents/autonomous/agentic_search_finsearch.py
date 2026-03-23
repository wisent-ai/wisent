from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

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
    evaluator_name = "finsearchcomp"

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
        *,
        oversampling_factor: float | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from FinSearchComp dataset.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        from wisent.core.utils.config_tools.constants import EXTRACTOR_DEFAULT_OVERSAMPLING_FACTOR
        max_items = self._normalize_limit(limit)
        if oversampling_factor is None:
            oversampling_factor = EXTRACTOR_DEFAULT_OVERSAMPLING_FACTOR
        pairs: list[ContrastivePair] = []

        try:
            # Load extra docs since some have null response_reference
            load_limit = int(max_items * oversampling_factor) if max_items else None
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
                load_limit = int(max_items * oversampling_factor) if max_items else None
                docs = self.load_dataset(
                    dataset_name="ByteSeedXpert/FinSearchComp",
                    split="train",
                    limit=load_limit,
                )
                log.info(f"Loaded {len(docs)} examples from FinSearchComp (train)")
            except Exception as e2:
                # Try validation split
                try:
                    load_limit = int(max_items * oversampling_factor) if max_items else None
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

