from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["SWEBenchVerifiedExtractor"]

log = setup_logger(__name__)


class SWEBenchVerifiedExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SWE-bench Verified - human-validated software engineering benchmark.

    SWE-bench Verified is a subset of 500 samples from SWE-bench that have been
    human-validated for quality. It evaluates LLMs on real-world software engineering
    tasks from popular Python repositories.

    Schema (princeton-nlp/SWE-bench_Verified):
        - repo: str (repository identifier)
        - instance_id: str (formatted instance identifier)
        - problem_statement: str (issue description)
        - patch: str (gold solution patch)
        - test_patch: str (test patch)
        - base_commit: str (commit hash before solution)
        - hints_text: str (hints/comments)
        - FAIL_TO_PASS: str (tests that should go from fail to pass)
        - PASS_TO_PASS: str (tests that should remain passing)
        - difficulty: str (difficulty rating)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "code_generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SWE-bench Verified examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} SWE-bench Verified examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SWE-bench Verified pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            repo = doc.get("repo", "")
            instance_id = doc.get("instance_id", "")
            problem_statement = doc.get("problem_statement", "").strip()
            patch = doc.get("patch", "").strip()
            hints_text = doc.get("hints_text", "")
            difficulty = doc.get("difficulty", "unknown")

            if not problem_statement or not patch:
                log.debug("Skipping: missing problem_statement or patch")
                return None

            # Format the prompt as a software engineering task
            formatted_question = self._format_prompt(repo, problem_statement, hints_text)

            # The correct answer is the gold patch
            correct_answer = patch

            # Create an incorrect answer (a patch that doesn't solve the issue)
            incorrect_answer = self._create_incorrect_patch(patch, problem_statement)

            metadata = {
                "label": "swe_bench_verified",
                "source": "princeton-nlp/SWE-bench_Verified",
                "repo": repo,
                "instance_id": instance_id,
                "difficulty": difficulty,
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

    def _format_prompt(self, repo: str, problem_statement: str, hints_text: str) -> str:
        """Format the SWE-bench problem as a prompt."""
        prompt_parts = [
            f"You are a software engineer working on the {repo} repository.",
            "",
            "## Issue Description",
            problem_statement,
        ]

        if hints_text and hints_text.strip():
            prompt_parts.extend([
                "",
                "## Additional Context",
                hints_text,
            ])

        prompt_parts.extend([
            "",
            "## Task",
            "Write a patch that resolves this issue. Provide the patch in unified diff format.",
            "",
            "## Patch:",
        ])

        return "\n".join(prompt_parts)

    def _create_incorrect_patch(self, correct_patch: str, problem_statement: str) -> str:
        """Create an incorrect patch that doesn't solve the issue."""
        # Strategy 1: Empty or minimal patch - return incomplete placeholder
        return """--- a/placeholder.py
+++ b/placeholder.py
@@ -1,3 +1,4 @@
 # File placeholder
+# TODO: This issue needs further investigation
 pass
"""

