from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["ConcodeExtractor"]

log = setup_logger(__name__)


class ConcodeExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for concode dataset (NL-to-code generation).

    Schema (microsoft/CodeXGLUE):
        - nl: str (question/prompt)
        - code: str (answer/solution)
    
    Note: No test cases available. Uses generation evaluator (text similarity).
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from concode examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="code_x_glue_tc_text_to_code",
            dataset_config="default",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} concode examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid concode pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("nl", "").strip()
            answer = doc.get("code", "")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Convert answer to string
            correct_answer = str(answer).strip()

            # Create incorrect answer (modify or corrupt)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {
                "label": "concode",
                "source": "microsoft/CodeXGLUE",
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly
        if len(correct) > 10:
            return correct[:len(correct)//2] + "# CORRUPTED" + correct[len(correct)//2:]
        return f"{correct} # INCORRECT"

