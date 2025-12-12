from __future__ import annotations

import json
import random
import re
from typing import Any

from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AppsExtractor"]

log = setup_logger(__name__)


class AppsExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for apps dataset.

    Schema (codeparrot/apps):
        - question: str (question/prompt)
        - solutions: str (answer/solution)
    """

    evaluator_name = "coding"

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

        # Load dataset (apps requires a configuration)
        docs = self.load_dataset(
            dataset_name="codeparrot/apps",
            dataset_config="all",
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

            # Prepend common imports (APPS solutions assume LeetCode-style environment)
            correct_answer = self._prepend_imports(correct_answer)

            # Create incorrect answer (modify or corrupt)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            # Parse input_output JSON to create test code
            test_code = None
            entry_point = None
            if input_output:
                try:
                    io_data = json.loads(input_output) if isinstance(input_output, str) else input_output
                    test_code, entry_point = self._build_test_code_from_io(io_data)
                except (json.JSONDecodeError, TypeError) as e:
                    log.debug(f"Could not parse input_output: {e}")

            metadata = {
                "label": "apps",
                "source": "codeparrot/apps",
                "test_code": test_code,
                "entry_point": entry_point,
                "language": "python",
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

    @staticmethod
    def _build_test_code_from_io(io_data: dict) -> tuple[str, str | None]:
        """Build test code from input/output data.

        APPS has two types of problems:
        1. stdin/stdout: No fn_name, run via subprocess
        2. call-based: Has fn_name, import and call Solution().fn_name()

        Returns:
            Tuple of (test_code, entry_point)
        """
        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])
        fn_name = io_data.get("fn_name")

        if not inputs or not outputs:
            return None, None

        if fn_name:
            return AppsExtractor._build_call_based_test_code(inputs, outputs, fn_name)
        else:
            return AppsExtractor._build_stdin_test_code(inputs, outputs)

    @staticmethod
    def _build_call_based_test_code(
        inputs: list, outputs: list, fn_name: str
    ) -> tuple[str, None]:
        """Build test code for call-based (LeetCode-style) problems."""
        total = len(inputs)
        test_code = f'''import sys
from solution import Solution
from typing import List, Optional, Dict, Tuple, Set, Any

def compare_outputs(actual, expected):
    """Compare outputs, handling floating point and nested structures."""
    if isinstance(expected, float) and isinstance(actual, float):
        return abs(actual - expected) < 1e-6
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(compare_outputs(a, e) for a, e in zip(actual, expected))
    return actual == expected

if __name__ == '__main__':
    sol = Solution()
    passed = 0
    total = {total}
'''
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            # inp is typically a list of arguments
            if isinstance(inp, list):
                args_repr = ", ".join(repr(arg) for arg in inp)
            else:
                args_repr = repr(inp)
            test_code += f"    # Test case {i+1}\n"
            test_code += f"    try:\n"
            test_code += f"        result = sol.{fn_name}({args_repr})\n"
            test_code += f"        expected = {repr(out)}\n"
            test_code += f"        if compare_outputs(result, expected):\n"
            test_code += f"            passed += 1\n"
            test_code += f"    except Exception:\n"
            test_code += f"        pass\n\n"

        test_code += "    print(f'PASSED:{passed}/{total}')\n"
        test_code += "    sys.exit(0 if passed == total else 1)\n"
        return test_code, None

    @staticmethod
    def _build_stdin_test_code(inputs: list, outputs: list) -> tuple[str, None]:
        """Build test code for stdin/stdout style problems."""
        total = len(inputs)
        test_code = f'''import subprocess
import sys

def normalize_output(s):
    """Normalize output by stripping trailing whitespace from each line."""
    lines = s.split('\\n')
    normalized = '\\n'.join(line.rstrip() for line in lines)
    return normalized.strip()

def run_solution(input_str):
    """Run solution.py with given input and return output."""
    result = subprocess.run(
        [sys.executable, "solution.py"],
        input=input_str,
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError(f"Solution failed: {{result.stderr}}")
    return result.stdout

if __name__ == '__main__':
    passed = 0
    total = {total}
'''
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            test_code += f"    # Test case {i+1}\n"
            test_code += f"    try:\n"
            test_code += f"        result = run_solution({repr(inp)})\n"
            test_code += f"        expected = {repr(out)}\n"
            test_code += f"        if normalize_output(result) == normalize_output(expected):\n"
            test_code += f"            passed += 1\n"
            test_code += f"    except Exception:\n"
            test_code += f"        pass\n\n"

        test_code += "    print(f'PASSED:{passed}/{total}')\n"
        test_code += "    sys.exit(0 if passed == total else 1)\n"
        return test_code, None

    # Common imports for LeetCode-style solutions
    COMMON_IMPORTS = """\
from typing import List, Optional, Dict, Tuple, Set, Any
import collections
import heapq
import itertools
import functools
import math
import bisect
from collections import defaultdict, Counter, deque
"""

    @staticmethod
    def _prepend_imports(code: str) -> str:
        """Prepend common imports to solution code.

        APPS solutions assume LeetCode-style environment where
        List, collections, heapq, etc. are pre-imported.
        """
        # Skip if code already has typing imports
        if "from typing import" in code or "import typing" in code:
            return code
        return AppsExtractor.COMMON_IMPORTS + code

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by shuffling letters in words.

        This reliably breaks code by corrupting variable/function names,
        causing NameError or SyntaxError.
        """
        def shuffle_word(word: str) -> str:
            """Shuffle all letters in a word."""
            if len(word) <= 2:
                return word
            letters = list(word)
            random.shuffle(letters)
            shuffled = ''.join(letters)
            if shuffled == word:
                return word[::-1]  # Reverse if shuffle didn't change
            return shuffled

        def replace_word(match: re.Match) -> str:
            word = match.group(0)
            return shuffle_word(word)

        # Shuffle words with 3+ characters
        result = re.sub(r'[A-Za-z]{3,}', replace_word, correct)

        # If nothing changed (all short words), append syntax error
        if result == correct:
            result = correct + "\n!!SYNTAX_ERROR!!"

        return result

