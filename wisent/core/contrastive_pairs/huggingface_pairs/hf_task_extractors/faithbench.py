from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger
import json
import requests

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["FaithBenchExtractor"]

log = setup_logger(__name__)

# GitHub raw URLs for FaithBench data
FAITHBENCH_GITHUB_BASE = "https://raw.githubusercontent.com/vectara/FaithBench/main/data_for_release"
FAITHBENCH_BATCH_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]  # batch 13 doesn't exist

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

        Loads data from GitHub vectara/FaithBench repository.

        Creates pairs for hallucination detection:
        - Positive (correct) = Accurate detection of hallucination
        - Negative (incorrect) = Missed or false positive detection

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        # Load from GitHub JSON files
        docs = self._load_from_github(max_items)
        
        if not docs:
            log.error("Failed to load FaithBench data from GitHub")
            return []

        log.info(f"Loaded {len(docs)} examples from FaithBench GitHub")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid FaithBench pairs extracted")

        return pairs

    def _load_from_github(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load FaithBench data from GitHub repository."""
        all_samples = []
        
        for batch_id in FAITHBENCH_BATCH_IDS:
            if limit and len(all_samples) >= limit:
                break
                
            url = f"{FAITHBENCH_GITHUB_BASE}/batch_{batch_id}.json"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                batch_data = response.json()
                
                # Extract samples from batch
                samples = batch_data.get("samples", [])
                all_samples.extend(samples)
                
                log.debug(f"Loaded {len(samples)} samples from batch_{batch_id}")
                
            except Exception as e:
                log.warning(f"Failed to load batch_{batch_id}: {e}")
                continue
        
        return all_samples[:limit] if limit else all_samples

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

