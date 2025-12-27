from __future__ import annotations

import requests
from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AiderPolyglotExtractor"]

log = setup_logger(__name__)

# GitHub API base URL for Aider Polyglot benchmark
AIDER_GITHUB_API = "https://api.github.com/repos/Aider-AI/polyglot-benchmark/contents"

# Languages supported by Aider Polyglot benchmark
AIDER_POLYGLOT_LANGUAGES = ["python", "javascript", "java", "cpp", "go", "rust"]


class AiderPolyglotExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Aider Polyglot benchmark.

    GitHub: https://github.com/Aider-AI/polyglot-benchmark
    
    Aider's polyglot benchmark tests LLMs on 225 challenging Exercism coding
    exercises across C++, Go, Java, JavaScript, Python, and Rust.

    Structure per exercise:
    - .docs/instructions.md - problem description
    - .meta/example.py - reference solution
    - {name}_test.py - test cases

    For code editing:
    - Positive (correct) = Working solution from .meta/example.py
    - Negative (incorrect) = Buggy or incomplete solution
    """

    evaluator_name = "code_editing"

    def __init__(self, language: str = "python"):
        """
        Initialize Aider Polyglot extractor.

        Args:
            language: Target programming language (python, javascript, java, cpp, go, rust)
        """
        super().__init__()
        if language not in AIDER_POLYGLOT_LANGUAGES:
            raise ValueError(f"Language must be one of {AIDER_POLYGLOT_LANGUAGES}")
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Aider Polyglot GitHub repository.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        exercises = self._load_exercises_from_github()
        
        if not exercises:
            log.error("Failed to load exercises from Aider Polyglot GitHub")
            return []

        log.info(f"Loaded {len(exercises)} exercises from Aider Polyglot GitHub")

        for exercise in exercises:
            pair = self._extract_pair_from_exercise(exercise)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Aider Polyglot pairs extracted")

        return pairs

    def _load_exercises_from_github(self) -> list[dict[str, Any]]:
        """Load exercises from Aider Polyglot GitHub repository."""
        try:
            # Get list of exercises
            exercises_url = f"{AIDER_GITHUB_API}/{self.language}/exercises/practice"
            response = requests.get(exercises_url, timeout=30)
            response.raise_for_status()
            
            exercise_dirs = response.json()
            exercises = []
            
            for exercise_dir in exercise_dirs:
                if exercise_dir.get("type") != "dir":
                    continue
                    
                exercise_name = exercise_dir.get("name", "")
                exercise_path = exercise_dir.get("path", "")
                
                # Load instructions and solution
                exercise_data = self._load_exercise_data(exercise_name, exercise_path)
                if exercise_data:
                    exercises.append(exercise_data)
            
            return exercises
            
        except Exception as e:
            log.error(f"Failed to load exercises from GitHub: {e}")
            return []

    def _load_exercise_data(self, name: str, path: str) -> dict[str, Any] | None:
        """Load a single exercise's instructions and solution."""
        try:
            base_url = "https://raw.githubusercontent.com/Aider-AI/polyglot-benchmark/main"
            
            # Load instructions
            instructions_url = f"{base_url}/{path}/.docs/instructions.md"
            instructions_resp = requests.get(instructions_url, timeout=15)
            if instructions_resp.status_code != 200:
                return None
            instructions = instructions_resp.text
            
            # Load solution - file extension depends on language
            ext_map = {
                "python": "py", "javascript": "js", "java": "java",
                "cpp": "cpp", "go": "go", "rust": "rs"
            }
            ext = ext_map.get(self.language, "py")
            
            solution_url = f"{base_url}/{path}/.meta/example.{ext}"
            solution_resp = requests.get(solution_url, timeout=15)
            if solution_resp.status_code != 200:
                return None
            solution = solution_resp.text
            
            return {
                "name": name,
                "instructions": instructions,
                "solution": solution,
                "path": path,
            }
            
        except Exception as e:
            log.debug(f"Failed to load exercise {name}: {e}")
            return None

    def _extract_pair_from_exercise(self, exercise: dict[str, Any]) -> ContrastivePair | None:
        """Convert an exercise into a ContrastivePair."""
        try:
            name = exercise.get("name", "")
            instructions = exercise.get("instructions", "").strip()
            solution = exercise.get("solution", "").strip()

            if not instructions or not solution:
                return None

            prompt = f"""Coding Exercise: {name.replace('-', ' ').title()}

{instructions}

Please provide the complete implementation."""

            correct_response = f"```{self.language}\n{solution}\n```"
            incorrect_response = self._create_incorrect_response(solution)

            metadata = {
                "label": "aider_polyglot",
                "source": "Aider-AI/polyglot-benchmark",
                "exercise_name": name,
                "language": self.language,
                "is_code_benchmark": True,
                "is_code_editing_benchmark": True,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair: {exc}", exc_info=True)
            return None

    def _create_incorrect_response(self, solution: str) -> str:
        """Create an incorrect response with common bugs."""
        lines = solution.split("\n")

        if len(lines) > 3:
            middle_idx = len(lines) // 2
            buggy_lines = lines[:middle_idx] + ["    pass  # TODO: incomplete"] + lines[middle_idx+2:]
            buggy = "\n".join(buggy_lines)
        elif lines:
            buggy = f"{lines[0]}\n    pass  # Implementation incomplete"
        else:
            buggy = "pass  # No implementation"

        return f"```{self.language}\n{buggy}\n```"

