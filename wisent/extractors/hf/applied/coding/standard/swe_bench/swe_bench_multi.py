from __future__ import annotations

from typing import Any
from wisent.core.cli.cli_logger import setup_logger

from wisent.core.contrastive_pairs.pair import ContrastivePair
from wisent.extractors.hf._registry.manifest.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

class MultiSWEBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Multi-SWE-bench - multilingual software engineering benchmark.

    Multi-SWE-bench extends SWE-bench to cover 7 programming languages:
    Java, TypeScript, JavaScript, Go, Rust, C, and C++. Contains 1,632
    high-quality instances curated by 68 expert annotators.

    Dataset: ByteDance-Seed/Multi-SWE-bench

    Schema:
        - org: str (GitHub organization)
        - repo: str (repository name)
        - number: int (PR number)
        - title: str (PR title)
        - body: str (PR description)
        - fix_patch: str (code fix)
        - test_patch: str (test modifications)
        - instance_id: str (unique identifier)
        - run_result: dict (test execution results)

    For multilingual software engineering evaluation:
    - Positive (correct) = Patch that resolves the issue across languages
    - Negative (incorrect) = Incomplete or language-inappropriate patch
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "software_engineering_multilingual"

    def __init__(self, language: str | None = None):
        """
        Initialize Multi-SWE-bench extractor.

        Args:
            language: Optional filter for language (java, typescript, javascript, go, rust, c, cpp)
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Multi-SWE-bench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="ByteDance-Seed/Multi-SWE-bench",
                split="train",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from Multi-SWE-bench")
        except Exception as e:
            log.error(f"Failed to load Multi-SWE-bench: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by language if specified
            if self.language:
                instance_id = doc.get("instance_id", "")
                # Language is typically part of repo name or instance_id
                if self.language.lower() not in instance_id.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Multi-SWE-bench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            org = doc.get("org", "").strip()
            repo = doc.get("repo", "").strip()
            title = doc.get("title", "").strip()
            body = doc.get("body", "").strip()
            fix_patch = doc.get("fix_patch", "").strip()
            test_patch = doc.get("test_patch", "")
            instance_id = doc.get("instance_id", "").strip()
            number = doc.get("number", 0)

            if not body and not title:
                log.debug("Skipping: missing title and body")
                return None

            if not fix_patch:
                log.debug("Skipping: missing fix_patch")
                return None

            # Build the task prompt
            task_prompt = self._build_task_prompt(org, repo, title, body, number)

            # Positive = correct patch
            correct_response = self._create_correct_response(fix_patch)
            # Negative = incorrect patch
            incorrect_response = self._create_incorrect_response(title)

            # Detect language from repo or instance_id
            language = self._detect_language(repo, instance_id)

            metadata = {
                "label": "multi_swe_bench",
                "source": "ByteDance-Seed/Multi-SWE-bench",
                "org": org,
                "repo": repo,
                "instance_id": instance_id,
                "language": language,
                "pr_number": number,
                "is_software_engineering_benchmark": True,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _detect_language(self, repo: str, instance_id: str) -> str:
        """Detect programming language from repo name or instance_id."""
        combined = f"{repo} {instance_id}".lower()
        for lang in MULTI_SWE_BENCH_LANGUAGES:
            if lang in combined:
                return lang
        return "unknown"

    def _build_task_prompt(
        self,
        org: str,
        repo: str,
        title: str,
        body: str,
        number: int,
    ) -> str:
        """Build the software engineering task prompt."""
        parts = [
            f"Repository: {org}/{repo}",
            f"PR #{number}: {title}",
        ]

        if body:
            parts.append(f"\n## Description\n{body}")

        parts.append(
            "\n## Task\nProvide a patch that addresses this pull request. "
            "The patch should correctly implement the required changes."
        )

        return "\n".join(parts)

    def _create_correct_response(self, patch: str) -> str:
        """Create a response with the correct patch."""
        return (
            f"Here is the patch to address this PR:\n\n"
            f"```diff\n{patch}\n```\n\n"
            "This patch implements the required changes while maintaining "
            "code quality and test coverage."
        )

    def _create_incorrect_response(self, title: str) -> str:
        """Create an incorrect response."""
        return (
            "I looked at the PR but here's an incomplete attempt:\n\n"
            "```diff\n"
            "- // Original implementation\n"
            "+ // Attempted fix - needs more work\n"
            "+ throw new Error('Not implemented');\n"
            "```\n\n"
            "This solution is incomplete and doesn't properly address the requirements."
        )

