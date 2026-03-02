"""Advanced converters for FullBenchmarkDownloader (math, code, GPQA, HLE, MBPP)."""

import random
import re
from typing import Any, Dict, List

from wisent.core.constants import DISPLAY_TRUNCATION_MEDIUM, DISTRACTOR_NEARBY_MIN, DISTRACTOR_NEARBY_MAX, DISTRACTOR_MAX_COUNT


class AdvancedConvertersMixin:
    """Mixin providing advanced format conversion methods including code perturbation."""

    def _generate_math_distractors(self, correct_answer: str) -> List[str]:
        """Generate plausible incorrect answers for math problems."""
        # Extract final number from answer
        numbers = re.findall(r"\d+(?:\.\d+)?", correct_answer)
        if not numbers:
            return ["42", "0", "Cannot be determined"]

        final_number = float(numbers[-1])

        # Generate distractors
        distractors = []

        # Off-by-one errors
        distractors.append(str(int(final_number + 1)))
        distractors.append(str(int(final_number - 1)))

        # Calculation errors (common mistakes)
        distractors.append(str(int(final_number * 2)))
        distractors.append(str(int(final_number / 2)))

        # Random nearby numbers
        distractors.append(str(int(final_number + random.randint(DISTRACTOR_NEARBY_MIN, DISTRACTOR_NEARBY_MAX))))

        return distractors[:DISTRACTOR_MAX_COUNT]  # Return top 3

    def _convert_humaneval_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert HumanEval code generation format."""
        task_id = sample.get("task_id", "unknown")
        prompt = sample.get("prompt", "")
        canonical_solution = sample.get("canonical_solution", "")
        test = sample.get("test", "")
        entry_point = sample.get("entry_point", "")

        pairs = []

        # Create a contrastive pair with the coding prompt
        pairs.append(
            {
                "question": f"Complete this Python function:\n\n{prompt}",
                "correct_answer": canonical_solution,
                "incorrect_answer": "# Incorrect or incomplete implementation\npass",
                "metadata": {
                    "task_id": task_id,
                    "test_cases": test,
                    "entry_point": entry_point,
                    "benchmark_type": "humaneval",
                    "task_type": "code_completion",
                },
            }
        )

        return pairs

    def _convert_gpqa_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert GPQA format (Question, choice1-4, answer, plus rich metadata)."""
        question = sample.get("Question", "")
        choice1 = sample.get("choice1", "")
        choice2 = sample.get("choice2", "")
        choice3 = sample.get("choice3", "")
        choice4 = sample.get("choice4", "")
        answer = sample.get("answer", "")

        # Extract letter from answer format like "(A)" or "A"
        answer_match = re.search(r"[ABCD]", answer.upper())
        if not answer_match:
            return []

        answer_letter = answer_match.group()

        # Map answer letter to choice
        choices_map = {"A": choice1, "B": choice2, "C": choice3, "D": choice4}

        correct_answer = choices_map.get(answer_letter, "")
        if not correct_answer:
            return []

        # Create pairs with each incorrect option
        pairs = []
        for letter, choice in choices_map.items():
            if letter != answer_letter and choice:
                pairs.append(
                    {
                        "context": question,
                        "good_response": correct_answer,
                        "bad_response": choice,
                        "metadata": {
                            "answer_key": answer_letter,
                            "raw_answer": answer,
                            "benchmark_type": "gpqa",
                            "subdomain": sample.get("Subdomain", ""),
                            "high_level_domain": sample.get("High-level domain", ""),
                            "difficulty_estimate": sample.get("Writer's Difficulty Estimate", ""),
                            "expert_accuracy": sample.get("Expert Validator Accuracy", ""),
                            "explanation": sample.get("Explanation", "")[:DISPLAY_TRUNCATION_MEDIUM]
                            if sample.get("Explanation")
                            else "",  # Truncate long explanations
                        },
                    }
                )

        return pairs

    def _convert_hle_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert HLE format (question, answer, answer_type, category)."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        answer_type = sample.get("answer_type", "")
        category = sample.get("category", "")

        if not question or not answer:
            return []

        # Use the HLE extractor to get contrastive pairs
        from wisent.core.benchmarks import HLEExtractor

        try:
            extractor = HLEExtractor()
            contrastive_pair = extractor.extract_contrastive_pair(sample)

            if contrastive_pair:
                return [
                    {
                        "question": contrastive_pair["question"],
                        "good_response": contrastive_pair["correct_answer"],
                        "bad_response": contrastive_pair["incorrect_answer"],
                        "metadata": {
                            "answer_type": answer_type,
                            "category": category,
                            "raw_subject": sample.get("raw_subject", ""),
                            "benchmark_type": "hle",
                        },
                    }
                ]
            return []
        except Exception as e:
            print(f"         Warning: Error converting HLE sample: {e}")
            return []

    def _perturb_code_to_break(self, code: str) -> str:
        """
        Perturb correct code to make it broken/unable to execute at runtime.

        Introduces various types of bugs:
        - Syntax errors (missing colons, parentheses)
        - Runtime errors (undefined variables)
        - Logic errors (wrong operators)
        - Type errors (wrong return values)

        Args:
            code: Correct Python code

        Returns:
            Broken version of the code
        """
        lines = code.split('\n')
        if not lines:
            return "pass  # Broken code"

        # Choose a random perturbation strategy
        perturbation_type = random.choice([
            'remove_colon',
            'remove_return',
            'wrong_variable',
            'syntax_error',
            'wrong_operator',
            'incomplete_code'
        ])

        if perturbation_type == 'remove_colon':
            # Remove colons from function/if/for statements
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ['def ', 'if ', 'for ', 'while ', 'elif ', 'else:']):
                    lines[i] = line.replace(':', '')
                    break

        elif perturbation_type == 'remove_return':
            # Remove return statement to break function
            for i, line in enumerate(lines):
                if 'return ' in line:
                    lines[i] = line.replace('return ', '# return ')
                    break

        elif perturbation_type == 'wrong_variable':
            # Use undefined variable name
            for i, line in enumerate(lines):
                if '=' in line and 'def ' not in line:
                    lines[i] = line.replace('=', '= undefined_variable +')
                    break

        elif perturbation_type == 'syntax_error':
            # Add syntax error by removing closing parenthesis
            for i, line in enumerate(lines):
                if '(' in line and ')' in line:
                    lines[i] = line.replace(')', '', 1)
                    break

        elif perturbation_type == 'wrong_operator':
            # Change operators to break logic
            for i, line in enumerate(lines):
                if any(op in line for op in ['+', '-', '*', '/', '<', '>', '==']):
                    line = line.replace('+', '-', 1) if '+' in line else line
                    line = line.replace('<', '>', 1) if '<' in line else line
                    lines[i] = line
                    break

        elif perturbation_type == 'incomplete_code':
            # Return only first half of code to make it incomplete
            lines = lines[:max(1, len(lines) // 2)]
            lines.append("    # Incomplete implementation")

        return '\n'.join(lines)

    def _convert_mbpp_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MBPP/HumanEval code generation format (task_id, code, prompt, test)."""
        task_id = sample.get("task_id", "")
        code = sample.get("code", "")
        prompt = sample.get("prompt", "")
        test = sample.get("test", "")

        # For code generation tasks, we create contrastive pairs based on:
        # Correct: The reference code solution
        # Incorrect: Perturbed version with bugs that prevent runtime execution

        pairs = []

        # Generate incorrect code by perturbing the correct solution
        incorrect_code = self._perturb_code_to_break(code)

        # Create a contrastive pair with the coding prompt
        pairs.append(
            {
                "question": f"Write Python code to solve this problem:\n\n{prompt}",
                "correct_answer": code,
                "incorrect_answer": incorrect_code,
                "metadata": {
                    "task_id": task_id,
                    "test_cases": test,
                    "source_file": sample.get("source_file", ""),
                    "test_imports": sample.get("test_imports", ""),
                    "test_list": sample.get("test_list", []),
                    "benchmark_type": "mbpp",
                    "task_type": "code_generation",
                    "programming_language": "python",
                },
            }
        )

        return pairs
