from __future__ import annotations

import json
from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["MercuryExtractor"]

log = setup_logger(__name__)


class MercuryExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Mercury - code efficiency benchmark.

    Dataset: Elfsong/Mercury
    Paper: "Mercury: A Code Efficiency Benchmark for LLM Code Synthesis"
    
    Mercury evaluates code efficiency by comparing different solutions
    to the same problem based on runtime performance.

    Schema:
        - prompt: str (problem description)
        - solutions: list[dict] with runtime and solution code
        - test_cases: str (JSON with test inputs/outputs)
        - difficulty: str
    
    For code efficiency evaluation:
    - Positive (correct) = Fastest solution
    - Negative (incorrect) = Slowest solution
    """

    evaluator_name = "code_efficiency"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mercury examples.
        """
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="Elfsong/Mercury",
            split="eval",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} Mercury examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Mercury pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        
        Uses fastest vs slowest solution as correct vs incorrect.
        """
        try:
            prompt = doc.get("prompt", "").strip()
            solutions = doc.get("solutions", [])
            difficulty = doc.get("difficulty", "")
            slug_name = doc.get("slug_name", "")
            pretty_content = doc.get("pretty_content", [])

            if not prompt or not solutions or len(solutions) < 2:
                return None

            # Sort solutions by runtime (fastest first)
            # Runtime format is like "44ms", "36ms", etc.
            def parse_runtime(sol):
                runtime_str = sol.get("runtime", "999ms")
                try:
                    return int(runtime_str.replace("ms", ""))
                except:
                    return 999
            
            sorted_solutions = sorted(solutions, key=parse_runtime)
            
            fastest = sorted_solutions[0]
            slowest = sorted_solutions[-1]
            
            fastest_code = fastest.get("solution", "")
            slowest_code = slowest.get("solution", "")
            
            if not fastest_code or not slowest_code:
                return None

            # Use pretty_content if available for problem description
            problem_desc = pretty_content[0] if pretty_content else prompt

            formatted_question = f"""Code Efficiency Task:

{problem_desc}

Write an efficient Python solution."""

            metadata = {
                "label": "mercury",
                "source": "Elfsong/Mercury",
                "slug_name": slug_name,
                "difficulty": difficulty,
                "fastest_runtime": fastest.get("runtime", ""),
                "slowest_runtime": slowest.get("runtime", ""),
                "is_code_efficiency_benchmark": True,
            }

            return self._build_pair(
                question=formatted_question,
                correct=f"```python\n{fastest_code}\n```",
                incorrect=f"```python\n{slowest_code}\n```",
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Mercury pair: {exc}", exc_info=True)
            return None

