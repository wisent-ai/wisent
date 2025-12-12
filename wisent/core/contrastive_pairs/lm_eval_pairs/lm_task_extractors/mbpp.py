from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MBPPExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "mbpp",
    "mbpp_instruct",
    "mbpp_plus",
    "mbpp_plus_instruct",
)

class MBPPExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the MBPP (Mostly Basic Python Problems) benchmark."""


    evaluator_name = "coding"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MBPP docs.

        MBPP schema:
            - text: str (problem description)
            - code: str (correct solution)
            - test_list: list[str] (test cases)
            - test_setup_code: str (setup code for tests)
            - challenge_test_list: list[str] (additional tests)

        Args:
            lm_eval_task_data: lm-eval task instance for MBPP.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        task_name = getattr(lm_eval_task_data, "NAME", "mbpp")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc, task_name)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid MBPP pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], task_name: str) -> ContrastivePair | None:
        """
        Convert a single MBPP doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("task_id", "unknown"))

        try:
            text = str(doc.get("text", "")).strip()
            code = str(doc.get("code", "")).strip()
            test_list = doc.get("test_list", [])

            if not text or not code:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Correct solution is the provided code
            correct = code

            # Incorrect solution: return a placeholder or buggy implementation
            incorrect = "    return None  # Incomplete implementation"

            # Format tests (use first 3 if available)
            tests_str = "\n".join(test_list[:3]) if test_list else ""

            # Different prompt format for instruct vs base
            is_instruct = "instruct" in task_name.lower()
            if is_instruct:
                formatted_question = f"You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n{tests_str}"
            else:
                formatted_question = f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{tests_str}\n[BEGIN]\n"

            # Extract entry_point (function name) from first test assertion
            entry_point = None
            if test_list:
                match = re.search(r'assert\s+(\w+)\(', test_list[0])
                entry_point = match.group(1) if match else None

            # Format test_code with check() function (like HumanEval format)
            # Replace function name with 'candidate' in assertions
            if test_list and entry_point:
                # Convert "assert func_name(...)" to "assert candidate(...)"
                converted_tests = [
                    re.sub(rf'\b{entry_point}\b', 'candidate', test)
                    for test in test_list
                ]
                test_code = f"def check(candidate):\n    " + "\n    ".join(converted_tests)
            else:
                test_code = ""

            metadata = {
                "label": task_name,
                "entry_point": entry_point,
                "test_code": test_code,
                "language": "python",
                "task_name": task_name,
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
