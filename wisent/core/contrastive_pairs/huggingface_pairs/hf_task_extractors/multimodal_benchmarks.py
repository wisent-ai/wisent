"""Extractors for multimodal benchmarks."""
from __future__ import annotations

import random
from typing import Any

from wisent.core.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "MMMUExtractor",
]

log = setup_logger(__name__)


class MMMUExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MMMU - Massive Multi-discipline Multimodal Understanding.

    Dataset: MMMU/MMMU on HuggingFace

    MMMU is a multimodal benchmark designed to evaluate multimodal models on
    massive multi-discipline tasks requiring college-level subject knowledge
    and deliberate reasoning. Covers 30 subjects across 6 disciplines.

    Note: This extractor focuses on the text-only components of the questions
    since contrastive pair generation is text-based. The image information
    is stored in metadata for reference.
    """

    evaluator_name = "multiple_choice"

    def __init__(self, subject: str | None = None):
        """
        Initialize MMMU extractor.

        Args:
            subject: Optional subject filter (e.g., 'Art', 'Physics', 'Math')
        """
        super().__init__()
        self.subject = subject

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from MMMU dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # MMMU has multiple subjects as configs
            config = self.subject if self.subject else None
            docs = self.load_dataset(
                dataset_name="MMMU/MMMU",
                dataset_config=config,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from MMMU")
        except Exception as e:
            log.warning(f"Failed to load MMMU with config: {e}")
            # Try validation split
            try:
                docs = self.load_dataset(
                    dataset_name="MMMU/MMMU",
                    dataset_config=config,
                    split="validation",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from MMMU (validation)")
            except Exception as e2:
                log.error(f"Failed to load MMMU: {e2}")
                return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            question = doc.get("question", "").strip()
            options = doc.get("options", [])
            answer = doc.get("answer", "")

            # Get subject info
            subject = doc.get("subject", doc.get("subfield", "general"))

            if not question:
                return None

            # Handle options - they might be stored as list or individual fields
            if not options:
                # Try to get options from individual fields
                option_keys = ["option_a", "option_b", "option_c", "option_d", "option_e"]
                options = [doc.get(k, "") for k in option_keys if doc.get(k)]

            if not options or len(options) < 2:
                # If no options, this might be a free-form question
                task_prompt = f"""Question: {question}

Answer:"""
                correct = str(answer).strip() if answer else "A"
                incorrect = "I don't know"
            else:
                # Multiple choice format
                choice_letters = ['A', 'B', 'C', 'D', 'E']
                choices_text = "\n".join(
                    f"{choice_letters[i]}. {opt}" for i, opt in enumerate(options[:5])
                )

                task_prompt = f"""Question: {question}

{choices_text}

Answer:"""

                # Parse answer
                if isinstance(answer, int) and answer < len(options):
                    correct = choice_letters[answer]
                elif isinstance(answer, str):
                    answer_upper = answer.upper().strip()
                    if answer_upper in choice_letters:
                        correct = answer_upper
                    elif answer in options:
                        correct = choice_letters[options.index(answer)]
                    else:
                        # Answer might be the letter directly
                        correct = answer_upper[0] if answer_upper else "A"
                else:
                    correct = "A"

                # Get incorrect answer
                if correct in choice_letters:
                    correct_idx = choice_letters.index(correct)
                    wrong_indices = [i for i in range(len(options)) if i != correct_idx]
                    incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"
                else:
                    incorrect = "B"

            # Check for image references (for metadata, not for extraction)
            has_image = bool(doc.get("image") or doc.get("image_1") or doc.get("images"))

            metadata = {
                "label": "mmmu",
                "source": "MMMU/MMMU",
                "subject": subject,
                "has_image": has_image,
                "is_multimodal_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting MMMU pair: {exc}", exc_info=True)
            return None
