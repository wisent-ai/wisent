from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["HendrycksMathExtractor"]
_LOG = setup_logger(__name__)


class HendrycksMathExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Hendrycks Math benchmark and all its subtasks."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Hendrycks Math docs.

        Args:
            lm_eval_task_data: lm-eval task instance.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            # Use task_data.doc_to_text for formatted question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                # Fallback: try to get problem field
                formatted_question = doc.get("problem", doc.get("question", str(doc)))

            # Get the solution - hendrycks_math uses "solution" field with full explanation
            solution = doc.get("solution", "")

            if not formatted_question or not solution:
                _LOG.debug("Skipping: missing question or solution")
                return None

            # Extract the final answer from \boxed{} notation
            correct_answer = self._extract_boxed_answer(solution)
            if not correct_answer:
                # If we can't extract from boxed notation, use the whole solution
                correct_answer = solution
                _LOG.debug("Could not extract boxed answer, using full solution")

            # Generate incorrect answer based on the extracted answer
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            task_name = getattr(task_data, "NAME", "hendrycks_math")
            metadata = {
                "label": task_name,
                "source": task_name,
            }

            return self._build_pair(
                question=formatted_question,
                correct=str(correct_answer),
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            _LOG.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    @staticmethod
    def _extract_boxed_answer(solution: str) -> str | None:
        """
        Extract the answer from LaTeX \\boxed{} notation.

        Args:
            solution: The full solution string containing \\boxed{answer}

        Returns:
            The extracted answer or None if not found
        """
        # Find \boxed{ and then match balanced braces
        start_pattern = r'\\boxed\{'
        match = re.search(start_pattern, solution)
        if not match:
            return None

        # Start after \boxed{
        start_idx = match.end()
        brace_count = 1
        idx = start_idx

        # Find the matching closing brace
        while idx < len(solution) and brace_count > 0:
            if solution[idx] == '{':
                brace_count += 1
            elif solution[idx] == '}':
                brace_count -= 1
            idx += 1

        if brace_count == 0:
            # Extract content between the braces
            return solution[start_idx:idx-1].strip()

        return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """
        Create an incorrect answer by modifying the correct one.

        Args:
            correct: The correct answer

        Returns:
            An incorrect answer
        """
        # Try to parse as number and modify it
        try:
            # Remove common LaTeX/math formatting
            clean = correct.replace('$', '').replace(',', '').replace('^\\circ', '').replace('^{\\circ}', '').strip()

            # Try integer
            num = int(clean)
            return str(num + 1)
        except ValueError:
            try:
                # Try float
                num = float(clean)
                return str(num + 1.0)
            except ValueError:
                # Can't parse as number, create a modified version
                # For fractions like \frac{8}{17}, modify numerator
                frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', correct)
                if frac_match:
                    num, denom = frac_match.groups()
                    return f"\\frac{{{int(num) + 1}}}{{{denom}}}"

                # For other cases, just append " + 1"
                return f"{correct} + 1"

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
