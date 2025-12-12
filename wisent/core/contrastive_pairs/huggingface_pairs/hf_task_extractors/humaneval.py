from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "HumanEvalUnifiedExtractor",
    "HumanEvalExtractor",
    "HumanEval64Extractor",
    "HumanEvalPlusExtractor",
    "HumanEvalInstructExtractor",
    "HumanEval64InstructExtractor",
]

log = setup_logger(__name__)

# Tasks supported by this extractor
task_names = (
    "humaneval",
    "humaneval_64",
    "humaneval_plus",
    "humaneval_instruct",
    "humaneval_64_instruct",
)


class HumanEvalUnifiedExtractor(HuggingFaceBenchmarkExtractor):
    """
    Unified extractor for all HumanEval variants.

    Supports:
        - humaneval, humaneval_64, humaneval_plus: raw prompt format
        - humaneval_instruct, humaneval_64_instruct: instruction format

    Dataset: openai_humaneval (164 Python problems)
    """

    evaluator_name = "coding"

    # Override in subclasses
    task_name = "humaneval"
    is_instruct = False

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
        **kwargs,
    ) -> list[ContrastivePair]:
        """Build contrastive pairs from HumanEval examples."""
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="openai_humaneval",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting {self.task_name} pairs from {len(docs)} examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning(f"No valid {self.task_name} pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single HumanEval doc into a ContrastivePair."""
        try:
            prompt = doc.get("prompt", "")
            body = doc.get("canonical_solution", "")
            entry_point = doc.get("entry_point", "")
            test_code = doc.get("test", "").strip()

            if not prompt or not body:
                log.debug("Skipping: missing prompt or solution")
                return None

            # Build complete function: prompt (signature + docstring) + body
            correct = prompt + body

            # Build incorrect answer: prompt + pass
            incorrect = self._create_incorrect_answer(prompt)

            # Format question based on task type just like lm eval harness does it
            formatted_question = self._format_question(prompt)

            metadata = {
                "label": self.task_name,
                "entry_point": entry_point,
                "test_code": test_code,
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _format_question(self, prompt: str) -> str:
        """Format the question based on task type."""
        if self.is_instruct:
            # lm_eval instruction format
            return (
                "Write me a solution to the following problem and make sure "
                "that it passes the tests:\n"
                f"```python\n{prompt}\n```"
            )
        else:
            # Raw prompt for base models
            return prompt


    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly
        if len(correct) > 10:
            return correct[:len(correct)//2] + "# CORRUPTED" + correct[len(correct)//2:]
        return f"{correct} # INCORRECT"


# Subclasses for each task variant

class HumanEvalExtractor(HumanEvalUnifiedExtractor):
    task_name = "humaneval"
    is_instruct = False


class HumanEval64Extractor(HumanEvalUnifiedExtractor):
    task_name = "humaneval_64"
    is_instruct = False


class HumanEvalPlusExtractor(HumanEvalUnifiedExtractor):
    task_name = "humaneval_plus"
    is_instruct = False


class HumanEvalInstructExtractor(HumanEvalUnifiedExtractor):
    task_name = "humaneval_instruct"
    is_instruct = True


class HumanEval64InstructExtractor(HumanEvalUnifiedExtractor):
    task_name = "humaneval_64_instruct"
    is_instruct = True
