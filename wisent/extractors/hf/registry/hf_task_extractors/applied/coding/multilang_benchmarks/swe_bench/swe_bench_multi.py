from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

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
    evaluator_name = "multi_swe_bench"

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

        # The full dataset has python file empty + some files fail generation;
        # load specific language jsonls directly via hf_hub_download
        try:
            from huggingface_hub import hf_hub_download
            import json

            # Try multiple language files since python is empty in this repo
            file_candidates = [
                "java/alibaba__fastjson2_dataset.jsonl",
                "go/cli__cli_dataset.jsonl",
                "js/expressjs__express_dataset.jsonl",
                "rust/BurntSushi__ripgrep_dataset.jsonl",
                "c/jqlang__jq_dataset.jsonl",
                "cpp/fmtlib__fmt_dataset.jsonl",
            ]
            docs: list[dict[str, Any]] = []
            for candidate in file_candidates:
                try:
                    jsonl_path = hf_hub_download(
                        repo_id="ByteDance-Seed/Multi-SWE-bench",
                        filename=candidate,
                        repo_type="dataset",
                    )
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                docs.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                            if max_items and len(docs) >= max_items * 2:
                                break
                except Exception:
                    continue
                if max_items and len(docs) >= max_items * 2:
                    break
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
            org = (doc.get("org") or "").strip()
            repo = (doc.get("repo") or "").strip()
            title = (doc.get("title") or "").strip()
            body = (doc.get("body") or "").strip()
            fix_patch = (doc.get("fix_patch") or "").strip()
            test_patch = doc.get("test_patch") or ""
            instance_id = (doc.get("instance_id") or "").strip()
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
        for lang in ("java", "typescript", "javascript", "go", "rust", "c", "cpp", "python"):
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

