from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger

__all__ = ["Ds1000Extractor"]
log = setup_logger(__name__)

task_names = ("ds1000",)


class Ds1000Extractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for DS-1000 Data Science coding benchmark.

    DS-1000 schema (xlangai/DS-1000):
        - prompt: str (problem description with code context)
        - reference_code: str (correct solution)
        - metadata: dict (library info, problem_id, etc.)
    """

    evaluator_name = "coding"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from DS-1000 examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="xlangai/DS-1000",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} DS-1000 examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid DS-1000 pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single DS-1000 doc into a ContrastivePair."""
        try:
            prompt = doc.get("prompt", "").strip()
            reference_code = doc.get("reference_code", "").strip()
            code_context = doc.get("code_context", "").strip()
            metadata_info = doc.get("metadata", {})
            
            if not prompt or not reference_code:
                return None

            question = f"Complete the following data science code:\n\n{prompt}"
            correct_code = reference_code
            incorrect_code = "result = None  # Incorrect: no implementation"

            # Build test code from code_context
            # DS-1000 has generate_test_case, exec_test, exec_context in code_context
            test_code = self._build_test_code(code_context, prompt)

            metadata = {
                "label": "ds1000",
                "problem_id": metadata_info.get("problem_id", ""),
                "library": metadata_info.get("library", ""),
                "test_code": test_code,
            }

            return self._build_pair(
                question=question,
                correct=correct_code,
                incorrect=incorrect_code,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_test_code(self, code_context: str, prompt: str) -> str | None:
        """Build executable test code from DS-1000 code_context.
        
        DS-1000 code_context contains:
        - generate_test_case(test_case_id): generates test input and expected result
        - exec_test(result, ans): checks if result matches expected
        - exec_context: template with [insert] placeholder
        - test_execution(): the actual test runner
        
        We use test_execution() directly which handles everything.
        """
        if not code_context:
            return None
        
        # DS-1000's code_context already has test_execution() that:
        # 1. Uses exec_context template with [insert] placeholder
        # 2. Generates test cases
        # 3. Executes and validates
        # We just need to call test_execution() with the solution code
        test_code = f'''{code_context}

# Read solution from file
solution_code = open("solution.py").read()

# Run DS-1000's built-in test execution
test_execution(solution_code)
print("Test passed!")
'''
        return test_code

