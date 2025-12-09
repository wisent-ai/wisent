from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["MultipleJavaExtractor"]

log = setup_logger(__name__)


class MultipleJavaExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MultiPL-E JavaScript benchmark.

    Uses HumanEval Python solutions as reference (generation evaluator).
    MultiPL-E doesn't include solutions, so we map problems to original HumanEval.
    """

    evaluator_name = "generation"

    def __init__(self):
        super().__init__()
        self._humaneval_solutions = None

    def _load_humaneval_solutions(self) -> dict[str, dict]:
        """Load HumanEval solutions indexed by problem number."""
        if self._humaneval_solutions is not None:
            return self._humaneval_solutions
        
        docs = self.load_dataset(
            dataset_name="openai_humaneval",
            split="test",
            limit=None,
        )
        
        self._humaneval_solutions = {}
        for doc in docs:
            task_id = doc.get("task_id", "")
            if "/" in task_id:
                problem_num = task_id.split("/")[1]
                self._humaneval_solutions[problem_num] = {
                    "prompt": doc.get("prompt", ""),
                    "canonical_solution": doc.get("canonical_solution", ""),
                    "entry_point": doc.get("entry_point", ""),
                }
        
        return self._humaneval_solutions

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="nuprl/MultiPL-E",
            dataset_config="humaneval-java",
            split="test",
            limit=max_items,
        )
        
        humaneval = self._load_humaneval_solutions()

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} multiple_java examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, humaneval)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid multiple_java pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], humaneval: dict) -> ContrastivePair | None:
        try:
            name = doc.get("name", "")
            js_prompt = doc.get("prompt", "").strip()
            tests = doc.get("tests", "")
            
            if not js_prompt:
                return None
            
            # Extract problem number from name like "HumanEval_0_has_close_elements"
            parts = name.split("_")
            if len(parts) >= 2 and parts[0] == "HumanEval":
                problem_num = parts[1]
            else:
                log.debug(f"Could not parse problem number from {name}")
                return None
            
            # Get original HumanEval solution
            he_data = humaneval.get(problem_num)
            if not he_data:
                log.debug(f"No HumanEval solution for problem {problem_num}")
                return None
            
            # Use Python solution as the "correct" concept
            correct_solution = he_data["prompt"] + he_data["canonical_solution"]
            
            # Create incorrect solution
            incorrect_solution = he_data["prompt"] + "\n    return None  # Incorrect implementation"

            metadata = {
                "label": "multiple_java",
                "source": "nuprl/MultiPL-E",
                "name": name,
                "language": "java",
                "problem_num": problem_num,
            }

            return self._build_pair(
                question=js_prompt,
                correct=correct_solution,
                incorrect=incorrect_solution,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None
