from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger
import requests
import zipfile
import json
import io

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.infra_tools.errors import InvalidValueError

log = setup_logger(__name__)

# GitHub URL for SciCode data
SCICODE_GITHUB_URL = "https://raw.githubusercontent.com/scicode-bench/scicode-bench.github.io/main/data/data.zip"

class NL2BashExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for NL2Bash - Natural Language to Bash command generation.

    Dataset: jiacheng-ye/nl2bash on HuggingFace
    
    NL2Bash evaluates LLMs' ability to translate natural language descriptions
    into correct Bash shell commands. Tests command syntax, flag usage,
    and understanding of CLI tools.

    For bash command generation evaluation:
    - Positive (correct) = Correct bash command with proper syntax
    - Negative (incorrect) = Command with errors, wrong syntax, or missing parts
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "nl2bash"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from NL2Bash dataset.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="jiacheng-ye/nl2bash",
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from nl2bash")
        except Exception as e:
            log.error(f"Failed to load nl2bash dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid NL2Bash pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.
        
        nl2bash schema:
        - nl: str (natural language description)
        - bash: str (correct bash command)
        """
        try:
            nl = doc.get("nl", "").strip()
            correct = doc.get("bash", "").strip()

            if not nl or not correct:
                return None

            task_prompt = f"""Bash Command Task:

{nl}

Provide the correct bash command to accomplish this task."""

            # Create incorrect by corrupting the command
            incorrect = self._create_incorrect_command(correct)

            correct_response = f"```bash\n{correct}\n```"
            incorrect_response = f"```bash\n{incorrect}\n```"

            metadata = {
                "label": "nl2bash",
                "source": "jiacheng-ye/nl2bash",
                "is_bash_benchmark": True,
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

    def _create_incorrect_command(self, correct: str) -> str:
        """Create a plausible but incorrect command by corrupting the correct one."""
        # Remove a flag or part of the command
        parts = correct.split()
        if len(parts) > 2:
            return " ".join(parts[:-1])  # Remove last part
        return correct + " --invalid-flag"



class SciCodeExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SciCode - scientific computing code generation benchmark.

    GitHub: https://scicode-bench.github.io/
    Paper: "SciCode: A Research Coding Benchmark Curated by Scientists"

    SciCode evaluates LLMs' ability to generate code for scientific computing
    tasks across Physics, Math, Material Science, Biology, and Chemistry.
    Contains 338 subproblems from 80 main challenges.

    For scientific computing evaluation:
    - Positive (correct) = Scientifically accurate code with proper numerical methods
    - Negative (incorrect) = Code with numerical errors or incorrect scientific methods
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "scicode"

    def __init__(self, http_timeout: int = 60, domain: str | None = None):
        """
        Initialize SciCode extractor.

        Args:
            http_timeout: Timeout in seconds for HTTP requests.
        Args:
            domain: Optional filter for scientific domain (physics, chemistry, biology, etc.)
        """
        super().__init__()
        self.http_timeout = http_timeout
        self.domain = domain

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SciCode examples.

        Loads data from GitHub ZIP archive.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        docs = self._load_from_github()
        
        if not docs:
            log.error("Failed to load SciCode data from GitHub")
            return []

        log.info(f"Loaded {len(docs)} problems from SciCode GitHub")

        for doc in docs:
            # Filter by domain if specified
            if self.domain and doc.get("domain", "").lower() != self.domain.lower():
                continue
                
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SciCode pairs extracted")

        return pairs

    def _load_from_github(self) -> list[dict[str, Any]]:
        """Load SciCode data from GitHub ZIP archive."""
        try:
            response = requests.get(SCICODE_GITHUB_URL, timeout=self.http_timeout)
            response.raise_for_status()
            
            all_problems = []
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for filename in zf.namelist():
                    if filename.endswith('.json'):
                        with zf.open(filename) as f:
                            try:
                                data = json.load(f)
                                if isinstance(data, list):
                                    all_problems.extend(data)
                                elif isinstance(data, dict):
                                    all_problems.append(data)
                            except json.JSONDecodeError:
                                continue
            
            return all_problems
            
        except Exception as e:
            log.error(f"Failed to load SciCode from GitHub: {e}")
            return []

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.
        
        SciCode schema varies by file, but typically includes:
        - problem_id: str
        - problem: str (description)
        - sub_problems: list of subproblems
        - domain: str (Physics, Math, etc.)
        """
        try:
            problem_id = doc.get("problem_id", "")
            problem = doc.get("problem", doc.get("description", "")).strip()
            domain = doc.get("domain", "general")
            sub_problems = doc.get("sub_problems", [])
            
            # Try to get problem text from various fields
            if not problem and sub_problems:
                problem = sub_problems[0].get("problem", "") if sub_problems else ""
            
            if not problem:
                return None

            task_prompt = f"""Scientific Computing Task ({domain}):

{problem}

Provide a Python implementation that is:
- Numerically accurate and stable
- Well-documented with clear variable names
- Efficient and follows scientific computing best practices"""

            # Create correct response placeholder (actual solution from benchmark)
            correct = doc.get("solution", doc.get("code", "# Correct solution would go here"))
            if isinstance(correct, list):
                correct = correct[0] if correct else "# Solution"
            
            # Create incorrect by corrupting
            incorrect = "# Incorrect implementation with numerical errors\nimport numpy as np\nresult = 0  # Wrong approach"

            correct_response = f"```python\n{correct}\n```"
            incorrect_response = f"```python\n{incorrect}\n```"

            metadata = {
                "label": "scicode",
                "source": "scicode-bench/SciCode",
                "problem_id": problem_id,
                "domain": domain,
                "is_scientific_computing_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting SciCode pair: {exc}", exc_info=True)
            return None

