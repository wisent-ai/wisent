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

    def _create_incorrect_answer(self, correct: str, doc: dict = None) -> str:
        """
        Create a meaningful incorrect answer by using different plausible wrong values.

        Strategy:
        1. For integers: use a different integer (multiply by 2, subtract, etc.)
        2. For fractions: change numerator/denominator in a plausible way
        3. For expressions: provide a structurally different but plausible answer

        Args:
            correct: The correct answer
            doc: Optional doc for context

        Returns:
            A plausible but incorrect answer
        """
        import random
        random.seed(hash(correct) % (2**32))  # Deterministic based on answer

        # Try to parse as number and create plausible wrong answer
        try:
            clean = correct.replace('$', '').replace(',', '').replace('^\\circ', '').replace('^{\\circ}', '').strip()

            # Try integer
            num = int(clean)
            # Use various wrong transformations
            wrong_transforms = [
                num * 2,           # doubled
                num // 2 if num > 1 else num * 3,  # halved or tripled
                num - 1 if num > 0 else num + 2,   # off by different amount
                num + 10,          # significantly different
                abs(num) * -1 if num > 0 else abs(num),  # sign flip
            ]
            return str(random.choice(wrong_transforms))
        except ValueError:
            try:
                # Try float
                num = float(clean)
                wrong_transforms = [
                    num * 2,
                    num / 2,
                    num - 0.5,
                    num + 0.25,
                    round(num) if num != round(num) else num + 0.5,
                ]
                return str(random.choice(wrong_transforms))
            except ValueError:
                pass

        # For fractions like \frac{8}{17}, create plausible wrong fraction
        frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', correct)
        if frac_match:
            num, denom = int(frac_match.group(1)), int(frac_match.group(2))
            wrong_fracs = [
                f"\\frac{{{denom}}}{{{num}}}",  # inverted
                f"\\frac{{{num}}}{{{denom + 1}}}",  # different denominator
                f"\\frac{{{num * 2}}}{{{denom}}}",  # doubled numerator
            ]
            return random.choice(wrong_fracs)

        # For sqrt expressions
        sqrt_match = re.search(r'\\sqrt\{(\d+)\}', correct)
        if sqrt_match:
            val = int(sqrt_match.group(1))
            wrong_vals = [val + 1, val - 1 if val > 1 else val + 2, val * 2]
            return correct.replace(f"\\sqrt{{{val}}}", f"\\sqrt{{{random.choice(wrong_vals)}}}")

        # For pi expressions
        if '\\pi' in correct:
            if '2\\pi' in correct:
                return correct.replace('2\\pi', '\\pi')
            elif '\\pi' in correct:
                return correct.replace('\\pi', '2\\pi')

        # For other symbolic answers, provide common wrong alternatives
        common_wrong = ['0', '1', '-1', '2', '\\infty', 'undefined']
        if correct not in common_wrong:
            return random.choice([w for w in common_wrong if w != correct])

        return "incorrect"

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
