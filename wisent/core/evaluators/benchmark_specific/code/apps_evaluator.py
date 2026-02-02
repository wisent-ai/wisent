"""
APPS (Automated Programming Progress Standard) Evaluator.

Script workflow:
1. code = APPSEvaluator.extract_code_from_json(raw_response)
2. code = AppsExtractor.prepend_imports(code)
3. test_code, _ = AppsExtractor.build_test_code(input_output)
4. result = evaluator.evaluate(code, expected=None, test_code=test_code)

Dataset: codeparrot/apps (10,000 Python coding problems)
Paper: https://arxiv.org/abs/2105.09938
"""

from __future__ import annotations

import json
import re
from typing import Any

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import CodingEvaluator

# Import shared utilities from extractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.apps import AppsExtractor


class APPSEvaluator(BaseEvaluator):
    """Evaluator for APPS benchmark. Delegates to CodingEvaluator."""

    name = "apps"
    description = "APPS coding benchmark evaluator"

    def __init__(self):
        self._coding_evaluator = CodingEvaluator()

    # --- Static helper methods for scripts ---
    # Reuse from AppsExtractor
    prepend_imports = staticmethod(AppsExtractor._prepend_imports)
    build_test_code = staticmethod(AppsExtractor._build_test_code_from_io)

    @staticmethod
    def get_prompt(
        question: str,
        starter_code: str | None = None,
        fn_name: str | None = None,
    ) -> str:
        """Generate prompt for the model.

        Args:
            question: Problem description
            starter_code: Optional starter code template
            fn_name: If provided, this is a call-based (LeetCode) problem

        Returns:
            Formatted prompt string
        """
        prompt = "You are an expert Python programmer. Solve the following coding problem.\n\n"
        prompt += f"Problem:\n{question}\n\n"

        if starter_code and starter_code.strip():
            prompt += f"Starter code (you must use this):\n```python\n{starter_code}\n```\n\n"

        prompt += 'Output your solution as a JSON object: {"code": "your_python_code_here"}\n'

        if fn_name:
            prompt += "Implement the Solution class with the required method.\n"
        else:
            prompt += "Your code should read from stdin and write to stdout.\n"

        prompt += "\nRespond with ONLY the JSON object, no other text."
        return prompt

    @staticmethod
    def extract_code_from_json(response: str) -> str | None:
        """Extract code from JSON response."""
        # Try direct JSON parse
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict) and "code" in data:
                return data["code"]
        except json.JSONDecodeError:
            pass

        # Try regex for multiline JSON
        match = re.search(r'\{\s*"code"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if "code" in data:
                    return data["code"]
            except json.JSONDecodeError:
                pass

        # Fallback: code blocks
        for pattern in [r'```python\n(.*?)\n```', r'```\n(.*?)\n```']:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1)

        return None

    # --- Evaluator interface ---

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate code against test cases.

        Args:
            response: Already-extracted and prepared Python code
            expected: Not used
            **kwargs:
                test_code: Already-generated test code

        Returns:
            EvalResult from CodingEvaluator
        """
        test_code = kwargs.get("test_code")

        if not test_code:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="No test_code provided",
            )

        return self._coding_evaluator.evaluate(
            response=response,
            expected=None,
            test_code=test_code,
            entry_point=None,
            task_name="apps",
            language="python",
        )
