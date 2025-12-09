from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["RecodeExtractor"]

log = setup_logger(__name__)


class RecodeExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for recode dataset (code search/retrieval).

    Schema (ARR-ADAPT/recode):
        - source: str (question/prompt)
        - target: str (answer/solution)
    
    Note: This is a code search task, not code execution. Uses generation evaluator.
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from recode examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset - using code_x_glue as alternative since ARR-ADAPT/recode doesn't exist
        docs = self.load_dataset(
            dataset_name="code_x_glue_tc_nl_code_search_adv",
            dataset_config="default",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} recode examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid recode pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            # code_x_glue_tc_nl_code_search_adv uses 'docstring' and 'code' fields
            question = doc.get("docstring", doc.get("source", "")).strip()
            answer = doc.get("code", doc.get("target", ""))

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
                "label": "recode",
                "source": "ARR-ADAPT/recode",
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
        # For code, corrupt the function name/signature (before first period)
        # This ensures the first sentence extraction will be different
        if len(correct) > 10:
            # Find the function definition line
            lines = correct.split('\n')
            if lines and 'def ' in lines[0]:
                # Corrupt the function name itself
                incorrect_lines = lines.copy()
                incorrect_lines[0] = incorrect_lines[0].replace('def ', 'def CORRUPTED_')
                incorrect = '\n'.join(incorrect_lines)

                # Verify correct is not still a substring of incorrect
                if correct in incorrect:
                    # Completely different function
                    incorrect = "def invalid_function():\n    '''This is intentionally wrong code'''\n    raise SyntaxError('Corrupted')"

                return incorrect
            else:
                # Not a function definition, use generic corruption
                incorrect = "# CORRUPTED CODE\n" + correct + "\n# REST IS INVALID"
                return incorrect

        return f"INVALID_{correct}"

