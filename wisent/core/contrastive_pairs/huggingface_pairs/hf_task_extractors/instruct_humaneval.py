from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["InstructHumanevalExtractor"]

log = logging.getLogger(__name__)


class InstructHumanevalExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for instruct_humaneval dataset.

    Schema (openai_humaneval):
        - prompt: str (question/prompt)
        - canonical_solution: str (answer/solution)
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from instruct_humaneval examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="openai_humaneval",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} instruct_humaneval examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid instruct_humaneval pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("prompt", "").strip()
            answer = doc.get("canonical_solution", "")
            test_code = doc.get("test", "").strip()
            entry_point = doc.get("entry_point", "")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # HumanEval canonical_solution is just the function body
            # We need to combine it with the function signature from the prompt
            # to create a complete executable function

            # First corrupt the function body BEFORE building the complete function
            # This ensures we corrupt the actual code, not the docstring
            correct_body = str(answer).strip()
            incorrect_body = self._create_incorrect_answer(correct_body)

            # Build complete functions with correct and incorrect bodies
            correct_answer = self._build_complete_function(question, correct_body)
            incorrect_answer = self._build_complete_function(question, incorrect_body)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {
                "label": "instruct_humaneval",
                "source": "openai_humaneval",
                "test_code": test_code,  # Include test code for execution
                "entry_point": entry_point,
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

    def _build_complete_function(self, prompt: str, function_body: str) -> str:
        """Build a complete function by combining prompt signature with function body.

        HumanEval prompt contains:
        - imports (e.g., from typing import List)
        - function signature (e.g., def func(args) -> ReturnType:)
        - docstring

        The canonical_solution contains only the function body (indented code).

        We need to combine them to create an executable function.
        """
        import re

        # Extract the function definition line (starts with 'def ')
        # This includes everything from 'def' up to and including the colon
        def_pattern = r'(def\s+\w+\([^)]*\)(?:\s*->\s*[^:]+)?:)'
        def_match = re.search(def_pattern, prompt, re.MULTILINE)

        if not def_match:
            # Fallback: return body as-is (might fail, but better than crashing)
            log.warning("Could not extract function signature from prompt")
            return function_body

        function_signature = def_match.group(1)

        # Extract any imports before the function definition
        imports_section = prompt[:def_match.start()].strip()

        # Extract docstring if present (text between the function signature and body)
        after_signature = prompt[def_match.end():].strip()
        docstring = ""
        if after_signature.startswith('"""') or after_signature.startswith("'''"):
            # Find the closing quotes
            quote_char = '"""' if after_signature.startswith('"""') else "'''"
            end_quote = after_signature.find(quote_char, len(quote_char))
            if end_quote != -1:
                docstring = "    " + after_signature[:end_quote + len(quote_char)]

        # Build the complete function
        parts = []
        if imports_section:
            parts.append(imports_section)
            parts.append("")  # Blank line after imports

        parts.append(function_signature)

        if docstring:
            parts.append(docstring)

        # Add the function body
        # The canonical_solution from HumanEval has its first line at column 0,
        # but subsequent lines are properly indented. We need to add 4 spaces
        # to the first line to match the indentation of other lines.
        if function_body:
            # Add indentation to the first line if it's not already indented
            body_lines = function_body.split('\n')
            if body_lines and body_lines[0] and not body_lines[0].startswith('    '):
                body_lines[0] = '    ' + body_lines[0]
            parts.append('\n'.join(body_lines))

        return "\n".join(parts)

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly by adding a comment in the middle
        if len(correct) > 10:
            return correct[:len(correct)//2] + "\n           # CORRUPTED" + correct[len(correct)//2:]
        return f"{correct} # INCORRECT"

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
