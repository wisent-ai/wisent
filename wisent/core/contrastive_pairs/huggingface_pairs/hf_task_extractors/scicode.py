from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["SciCodeExtractor"]

log = setup_logger(__name__)


class SciCodeExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SciCode - Scientific Computing Code Generation benchmark.

    SciCode evaluates LLMs on generating code for real scientific research problems
    across 16 subdomains from 5 domains: Physics, Math, Material Science, Biology,
    and Chemistry.

    The benchmark contains 338 subproblems from 80 main problems, with scientist-
    annotated gold-standard solutions.

    Schema (SciCode1/SciCode):
        - problem_name: str (name of the scientific problem)
        - problem_id: str (unique identifier)
        - problem_description_main: str (main problem description)
        - problem_background_main: str (background context)
        - problem_io: str (input/output specifications)
        - required_dependencies: str (required Python libraries)
        - sub_steps: List[dict] (solution substeps with ground truth)
        - general_solution: str (complete solution code)
        - general_tests: List (test cases)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "code_generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SciCode examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace - use test split
        docs = self.load_dataset(
            dataset_name="SciCode1/SciCode",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} SciCode examples")

        for doc in docs:
            # Each problem can have multiple sub-steps
            # We create pairs for both the main problem and sub-steps
            main_pair = self._extract_main_pair(doc)
            if main_pair is not None:
                pairs.append(main_pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

            # Also extract sub-step pairs if we have room
            if max_items is None or len(pairs) < max_items:
                sub_pairs = self._extract_substep_pairs(doc, max_items - len(pairs) if max_items else None)
                pairs.extend(sub_pairs)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SciCode pairs extracted")

        return pairs

    def _extract_main_pair(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Extract a contrastive pair from the main problem."""
        try:
            problem_name = doc.get("problem_name", "")
            problem_id = doc.get("problem_id", "")
            description = doc.get("problem_description_main", "").strip()
            background = doc.get("problem_background_main", "")
            problem_io = doc.get("problem_io", "")
            dependencies = doc.get("required_dependencies", "")
            solution = doc.get("general_solution", "").strip()

            if not description or not solution:
                log.debug("Skipping main problem: missing description or solution")
                return None

            # Format the prompt
            formatted_question = self._format_prompt(
                problem_name, description, background, problem_io, dependencies
            )

            # Create incorrect solution
            incorrect_solution = self._create_incorrect_solution(solution)

            metadata = {
                "label": "scicode",
                "source": "SciCode1/SciCode",
                "problem_id": problem_id,
                "problem_name": problem_name,
                "type": "main_problem",
            }

            return self._build_pair(
                question=formatted_question,
                correct=solution,
                incorrect=incorrect_solution,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting main pair: {exc}", exc_info=True)
            return None

    def _extract_substep_pairs(self, doc: dict[str, Any], limit: int | None) -> list[ContrastivePair]:
        """Extract contrastive pairs from sub-steps."""
        pairs = []
        sub_steps = doc.get("sub_steps", [])
        problem_id = doc.get("problem_id", "")
        problem_name = doc.get("problem_name", "")

        for step in sub_steps:
            if limit is not None and len(pairs) >= limit:
                break

            try:
                step_number = step.get("step_number", "")
                step_description = step.get("step_description_prompt", "").strip()
                step_background = step.get("step_background", "")
                ground_truth = step.get("ground_truth_code", "").strip()
                function_header = step.get("function_header", "")

                if not step_description or not ground_truth:
                    continue

                # Format sub-step prompt
                prompt_parts = [
                    f"## Scientific Computing Task: {problem_name} - Step {step_number}",
                    "",
                    "### Task Description",
                    step_description,
                ]

                if step_background:
                    prompt_parts.extend([
                        "",
                        "### Background",
                        step_background,
                    ])

                if function_header:
                    prompt_parts.extend([
                        "",
                        "### Function Signature",
                        f"```python\n{function_header}\n```",
                    ])

                prompt_parts.extend([
                    "",
                    "### Your Implementation",
                    "Write the Python code to solve this step:",
                ])

                formatted_question = "\n".join(prompt_parts)

                # Create incorrect solution
                incorrect_solution = self._create_incorrect_solution(ground_truth)

                metadata = {
                    "label": "scicode",
                    "source": "SciCode1/SciCode",
                    "problem_id": problem_id,
                    "problem_name": problem_name,
                    "step_number": step_number,
                    "type": "substep",
                }

                pair = self._build_pair(
                    question=formatted_question,
                    correct=ground_truth,
                    incorrect=incorrect_solution,
                    metadata=metadata,
                )
                pairs.append(pair)

            except Exception as exc:
                log.debug(f"Error extracting substep pair: {exc}")
                continue

        return pairs

    def _format_prompt(
        self,
        problem_name: str,
        description: str,
        background: str,
        problem_io: str,
        dependencies: str,
    ) -> str:
        """Format the main problem as a prompt."""
        prompt_parts = [
            f"## Scientific Computing Problem: {problem_name}",
            "",
            "### Problem Description",
            description,
        ]

        if background:
            prompt_parts.extend([
                "",
                "### Background",
                background,
            ])

        if problem_io:
            prompt_parts.extend([
                "",
                "### Input/Output Specification",
                problem_io,
            ])

        if dependencies:
            prompt_parts.extend([
                "",
                "### Required Dependencies",
                dependencies,
            ])

        prompt_parts.extend([
            "",
            "### Your Solution",
            "Write the complete Python code to solve this problem:",
        ])

        return "\n".join(prompt_parts)

    def _create_incorrect_solution(self, correct: str) -> str:
        """Create an incorrect solution that doesn't solve the problem."""
        # Strategy: Return an incomplete or placeholder implementation
        lines = correct.split("\n")

        # Find function definitions and return placeholder
        result_lines = []
        for line in lines:
            if line.strip().startswith("def "):
                result_lines.append(line)
                result_lines.append("    # TODO: implement this function")
                result_lines.append("    pass")
            elif line.strip().startswith("import ") or line.strip().startswith("from "):
                result_lines.append(line)

        if result_lines:
            return "\n".join(result_lines)

        # Fallback: return a stub
        return """# Placeholder implementation
def solve():
    # TODO: Implement the scientific computation
    raise NotImplementedError("Solution not implemented")
"""

