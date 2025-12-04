from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["FACTSGroundingExtractor"]

log = setup_logger(__name__)


class FACTSGroundingExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for FACTS Grounding - Factuality in Long-Form Responses (Google DeepMind 2025).

    FACTS Grounding evaluates LLMs' ability to generate responses that are
    factually accurate with respect to given context documents. Contains
    860 public examples across finance, law, medicine, and technology.

    Tasks include:
    - Summarization
    - Fact-finding
    - Comparative analysis

    Dataset: google/FACTS-grounding-public

    Schema:
        - system_instruction: str (system prompt)
        - user_request: str (user query)
        - context_document: str (source document, up to 32k tokens)
        - full_prompt: str (complete prompt)

    For factuality evaluation:
    - Positive (correct) = Response grounded in provided context
    - Negative (incorrect) = Response with hallucinations or unfounded claims
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "factuality_grounding"

    def __init__(self):
        """
        Initialize FACTS Grounding extractor.
        """
        super().__init__()

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from FACTS Grounding examples.

        Creates pairs for factuality evaluation:
        - Positive (correct) = Response grounded in context
        - Negative (incorrect) = Response with hallucinations

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="google/FACTS-grounding-public",
                dataset_config="examples",
                split="public",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from FACTS Grounding")
        except Exception as e:
            log.error(f"Failed to load FACTS Grounding: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid FACTS Grounding pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            system_instruction = doc.get("system_instruction", "").strip()
            user_request = doc.get("user_request", "").strip()
            context_document = doc.get("context_document", "").strip()

            if not user_request or not context_document:
                log.debug("Skipping: missing user_request or context_document")
                return None

            # Build the grounding task prompt
            task_prompt = self._build_grounding_prompt(
                system_instruction, user_request, context_document
            )

            # Positive = grounded response
            correct_response = self._create_grounded_response(
                user_request, context_document
            )
            # Negative = response with hallucinations
            incorrect_response = self._create_hallucinated_response(
                user_request, context_document
            )

            metadata = {
                "label": "facts_grounding",
                "source": "google/FACTS-grounding-public",
                "context_length": len(context_document),
                "is_factuality_benchmark": True,
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

    def _build_grounding_prompt(
        self,
        system_instruction: str,
        user_request: str,
        context: str,
    ) -> str:
        """Build the grounding task prompt."""
        parts = []

        if system_instruction:
            parts.append(f"System: {system_instruction}")

        parts.append(f"\nContext Document:\n{context}")
        parts.append(f"\nUser Request: {user_request}")
        parts.append(
            "\nProvide a response that is fully grounded in the provided context. "
            "Do not include information that is not supported by the document."
        )

        return "\n".join(parts)

    def _create_grounded_response(self, user_request: str, context: str) -> str:
        """Create a response that is properly grounded in the context."""
        return (
            f"Based on the provided document, I can address your request. "
            f"The information I'm sharing comes directly from the context provided. "
            f"[Response drawing from: '{context}']. "
            "All facts and claims in this response are supported by the source document, "
            "and I have not included any external information or assumptions not "
            "contained in the provided context."
        )

    def _create_hallucinated_response(self, user_request: str, context: str) -> str:
        """Create a response with hallucinated/ungrounded information."""
        return (
            f"Based on my knowledge and the document provided, here's what I found. "
            "According to industry experts and recent studies, this topic has been "
            "widely discussed. The latest research from 2024 suggests additional "
            "considerations that weren't mentioned in the document. Furthermore, "
            "based on common practices and my general understanding, I would "
            "recommend the following approach which goes beyond what's stated..."
        )

