from __future__ import annotations

import json
import requests
from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["RecodeExtractor"]

log = setup_logger(__name__)

# GitHub URL for ReCode dataset
RECODE_GITHUB_URL = "https://raw.githubusercontent.com/amazon-science/recode/main/dataset-release/nominal/HumanEval.jsonl"


class RecodeExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for ReCode - Robustness Evaluation of Code Generation Models.

    GitHub: https://github.com/amazon-science/recode
    Paper: "ReCode: Robustness Evaluation of Code Generation Models" (arXiv:2212.10264)
    
    ReCode evaluates code generation robustness using perturbed HumanEval/MBPP.
    The dataset includes:
    - Nominal (original) problems
    - Perturbed versions (docstring, function name, code syntax changes)

    Schema (HumanEval.jsonl):
        - task_id: str
        - prompt: str (function signature with docstring)
        - canonical_solution: str (reference solution)
        - test: str (test cases)
        - entry_point: str (function name)

    For robustness evaluation:
    - Positive (correct) = Canonical solution
    - Negative (incorrect) = Buggy/incomplete solution
    """

    evaluator_name = "code_generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from ReCode GitHub repository.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        docs = self._load_from_github()
        
        if not docs:
            log.error("Failed to load ReCode data from GitHub")
            return []

        log.info(f"Loaded {len(docs)} problems from ReCode GitHub")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid ReCode pairs extracted")

        return pairs

    def _load_from_github(self) -> list[dict[str, Any]]:
        """Load ReCode data from GitHub JSONL file."""
        try:
            response = requests.get(RECODE_GITHUB_URL, timeout=60)
            response.raise_for_status()
            
            problems = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    problems.append(json.loads(line))
            
            return problems
            
        except Exception as e:
            log.error(f"Failed to load ReCode from GitHub: {e}")
            return []

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            task_id = doc.get("task_id", "")
            prompt = doc.get("prompt", "").strip()
            canonical_solution = doc.get("canonical_solution", "").strip()
            entry_point = doc.get("entry_point", "")

            if not prompt or not canonical_solution:
                return None

            # Full correct code = prompt + solution
            correct_code = prompt + canonical_solution

            # Create incorrect by truncating/corrupting
            incorrect_code = self._create_incorrect_solution(prompt, canonical_solution)

            formatted_question = f"""Code Generation Task ({task_id}):

{prompt}

Complete the function implementation."""

            metadata = {
                "label": "recode",
                "source": "amazon-science/recode",
                "task_id": task_id,
                "entry_point": entry_point,
                "is_code_robustness_benchmark": True,
            }

            return self._build_pair(
                question=formatted_question,
                correct=f"```python\n{correct_code}\n```",
                incorrect=f"```python\n{incorrect_code}\n```",
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting ReCode pair: {exc}", exc_info=True)
            return None

    def _create_incorrect_solution(self, prompt: str, solution: str) -> str:
        """Create an incorrect solution by truncating or corrupting."""
        lines = solution.split('\n')
        
        if len(lines) > 2:
            # Truncate to first half + pass
            half = len(lines) // 2
            buggy = '\n'.join(lines[:half]) + '\n    pass  # incomplete'
        else:
            buggy = '    pass  # not implemented'
        
        return prompt + buggy

