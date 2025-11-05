from __future__ import annotations

from typing import Any
import logging
import json

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AppsExtractor"]

log = logging.getLogger(__name__)


class AppsExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for apps dataset.

    Schema (codeparrot/apps):
        - question: str (question/prompt)
        - solutions: str (answer/solution)
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from apps examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="codeparrot/apps",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} apps examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid apps pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("question", "").strip()
            answer = doc.get("solutions", "")
            input_output = doc.get("input_output", "")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Parse solutions JSON array and select one solution
            try:
                solutions_list = json.loads(answer) if isinstance(answer, str) else answer
                if not solutions_list or not isinstance(solutions_list, list):
                    log.debug("Skipping: solutions is not a valid list")
                    return None
                correct_answer = solutions_list[0].strip()
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                log.debug(f"Could not parse solutions array: {e}")
                return None

            # Create incorrect answer (modify or corrupt)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            # Parse input_output JSON to create test code
            test_code = None
            if input_output:
                try:
                    io_data = json.loads(input_output) if isinstance(input_output, str) else input_output
                    test_code = self._build_test_code_from_io(io_data)
                except (json.JSONDecodeError, TypeError) as e:
                    log.debug(f"Could not parse input_output: {e}")

            metadata = {
                "label": "apps",
                "source": "codeparrot/apps",
                "test_code": test_code,
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

    def _build_test_code_from_io(self, io_data: dict) -> str:
        """Build test code from input/output data."""
        # APPS input_output format: {"inputs": ["input1", "input2"], "outputs": ["output1", "output2"]}
        test_code = "def check(candidate):\n"
        test_code += "    import sys\n"
        test_code += "    from io import StringIO\n\n"

        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])

        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            test_code += f"    # Test case {i+1}\n"
            test_code += f"    old_stdin = sys.stdin\n"
            test_code += f"    sys.stdin = StringIO({repr(inp)})\n"
            test_code += f"    old_stdout = sys.stdout\n"
            test_code += f"    sys.stdout = StringIO()\n"
            test_code += f"    candidate()\n"
            test_code += f"    result = sys.stdout.getvalue()\n"
            test_code += f"    sys.stdin = old_stdin\n"
            test_code += f"    sys.stdout = old_stdout\n"
            test_code += f"    assert result.strip() == {repr(out)}.strip(), f'Expected {{repr({repr(out)})}}, got {{repr(result)}}'\n\n"

        return test_code if inputs and outputs else None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly
        if len(correct) > 10:
            return correct[:len(correct)//2] + "# CORRUPTED" + correct[len(correct)//2:]
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
