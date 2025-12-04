from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger
import json

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["FaithBenchExtractor"]

log = setup_logger(__name__)

# FaithBench hallucination categories
FAITHBENCH_CATEGORIES = [
    "Consistent",      # No hallucination
    "Questionable",    # Not clearly a hallucination
    "Benign",          # Hallucination but supported by world knowledge
    "Unwanted",        # Clear unwanted hallucination
]

# Unwanted hallucination subtypes
UNWANTED_SUBTYPES = [
    "Intrinsic",   # Contradicts the source
    "Extrinsic",   # Information not in source and not supported
]


class FaithBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for FaithBench - Summarization Hallucination Benchmark (2024).

    FaithBench evaluates faithfulness of LLM-generated summaries against
    source documents. Contains challenging hallucinations from 10 modern
    LLMs across 8 families with expert human annotations.

    Hallucination Categories:
    - Consistent: No hallucination detected
    - Questionable: Ambiguous cases
    - Benign: Hallucination supported by world knowledge
    - Unwanted: Clear harmful hallucinations (Intrinsic/Extrinsic)

    For hallucination detection:
    - Positive (correct) = Correctly identifies hallucination status
    - Negative (incorrect) = Incorrectly identifies hallucination status

    Data source: GitHub vectara/FaithBench repository
    Schema:
        - sample_id: int (unique identifier)
        - source: str (original document text)
        - summary: str (LLM-generated summary)
        - annotations: list[dict] (expert hallucination annotations)
        - metadata: dict (summarizer model, detector predictions)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "hallucination_detection"

    def __init__(self, include_benign: bool = False):
        """
        Initialize FaithBench extractor.

        Args:
            include_benign: If True, include benign hallucinations as positive examples
        """
        super().__init__()
        self.include_benign = include_benign

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from FaithBench examples.

        Creates pairs for hallucination detection:
        - Positive (correct) = Accurate detection of hallucination
        - Negative (incorrect) = Missed or false positive detection

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try to load from HuggingFace if available
        try:
            docs = self.load_dataset(
                dataset_name="vectara/FaithBench",
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from FaithBench HuggingFace")
        except Exception as e:
            log.warning(f"FaithBench not on HuggingFace, using synthetic examples: {e}")
            # Create synthetic examples based on FaithBench structure
            docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid FaithBench pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic examples based on FaithBench structure."""
        examples = []

        # Sample consistent (no hallucination) examples
        consistent_examples = [
            {
                "source": "The company reported quarterly revenue of $5.2 billion, up 12% from the previous year. The CEO attributed the growth to strong demand in the cloud computing division.",
                "summary": "The company's quarterly revenue reached $5.2 billion, representing a 12% year-over-year increase driven by cloud computing demand.",
                "has_hallucination": False,
                "category": "Consistent",
            },
            {
                "source": "Researchers at the university discovered a new species of deep-sea fish at depths of 3,000 meters. The fish has bioluminescent properties and measures approximately 15 centimeters in length.",
                "summary": "A new bioluminescent deep-sea fish species was discovered by university researchers at 3,000 meters depth, measuring about 15 cm.",
                "has_hallucination": False,
                "category": "Consistent",
            },
        ]

        # Sample unwanted hallucination examples
        unwanted_examples = [
            {
                "source": "The conference will take place in Boston from March 15-17. Registration opens January 1st and early bird pricing is available until February 1st.",
                "summary": "The conference is scheduled for March 15-17 in New York City. Registration begins January 1st with early bird discounts until February 1st.",
                "has_hallucination": True,
                "category": "Unwanted.Intrinsic",
                "hallucination_span": "New York City",
                "note": "Location changed from Boston to New York City",
            },
            {
                "source": "The study involved 500 participants across five countries over a two-year period. Results showed a 30% improvement in outcomes.",
                "summary": "The study with 500 participants from five countries over two years showed a 30% improvement. The lead researcher, Dr. Smith, plans further studies.",
                "has_hallucination": True,
                "category": "Unwanted.Extrinsic",
                "hallucination_span": "The lead researcher, Dr. Smith, plans further studies",
                "note": "No mention of Dr. Smith or future plans in source",
            },
        ]

        # Alternate between consistent and hallucinated examples
        for i in range(count):
            if i % 2 == 0:
                example = consistent_examples[i % len(consistent_examples)].copy()
            else:
                example = unwanted_examples[i % len(unwanted_examples)].copy()
            example["sample_id"] = i
            examples.append(example)

        return examples

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            sample_id = doc.get("sample_id", 0)
            source = doc.get("source", "").strip()
            summary = doc.get("summary", "").strip()
            annotations = doc.get("annotations", [])
            metadata_field = doc.get("metadata", {})

            if not source or not summary:
                log.debug("Skipping: missing source or summary")
                return None

            # Determine hallucination status
            has_hallucination = doc.get("has_hallucination", None)
            category = doc.get("category", "")

            if has_hallucination is None:
                # Infer from annotations
                if annotations:
                    # Has annotations = has hallucination
                    has_hallucination = True
                    # Get the most severe category
                    for annot in annotations:
                        label = annot.get("label", [])
                        if isinstance(label, list) and label:
                            category = label[0]
                        elif isinstance(label, str):
                            category = label
                else:
                    has_hallucination = False
                    category = "Consistent"

            # Skip benign hallucinations if not including them
            if not self.include_benign and "Benign" in category:
                return None

            # Build the detection task prompt
            task_prompt = self._build_detection_prompt(source, summary)

            if has_hallucination:
                correct_response = self._create_hallucination_detected_response(category, annotations)
                incorrect_response = self._create_no_hallucination_response()
            else:
                correct_response = self._create_no_hallucination_response()
                incorrect_response = self._create_false_positive_response()

            # Get summarizer model if available
            summarizer = metadata_field.get("summarizer", "") if isinstance(metadata_field, dict) else ""

            metadata = {
                "label": "faithbench",
                "source": "vectara/FaithBench",
                "sample_id": sample_id,
                "category": category,
                "has_hallucination": has_hallucination,
                "summarizer": summarizer,
                "is_hallucination_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_detection_prompt(self, source: str, summary: str) -> str:
        """Build the hallucination detection task prompt."""
        return f"""Evaluate whether the following summary contains any hallucinations compared to the source document.

A hallucination is information in the summary that:
- Contradicts the source document (Intrinsic hallucination)
- Adds information not present in or supported by the source (Extrinsic hallucination)

Source Document:
{source}

Summary to Evaluate:
{summary}

Does this summary contain hallucinations? Provide your assessment."""

    def _create_hallucination_detected_response(
        self, category: str, annotations: list[dict[str, Any]]
    ) -> str:
        """Create a response correctly identifying hallucination."""
        # Get specific details if available
        details = []
        for annot in annotations:
            span = annot.get("summary_span", "")
            note = annot.get("note", "")
            if span:
                details.append(f"'{span}'" + (f" - {note}" if note else ""))

        if "Intrinsic" in category:
            halluc_type = "intrinsic (contradicts source)"
        elif "Extrinsic" in category:
            halluc_type = "extrinsic (unsupported information)"
        else:
            halluc_type = "unwanted"

        response = f"Yes, this summary contains {halluc_type} hallucinations."
        if details:
            response += f" Specifically: {'; '.join(details)}"
        response += " The summary includes information that is either contradicted by or not present in the source document."

        return response

    def _create_no_hallucination_response(self) -> str:
        """Create a response indicating no hallucination."""
        return (
            "No, this summary is faithful to the source document. All information "
            "presented in the summary is accurately reflected in and supported by "
            "the source text. There are no contradictions or unsupported additions."
        )

    def _create_false_positive_response(self) -> str:
        """Create a false positive response (incorrectly detecting hallucination)."""
        return (
            "Yes, this summary appears to contain hallucinations. Some information "
            "seems inconsistent with or not directly supported by the source document."
        )

