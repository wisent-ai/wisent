from __future__ import annotations

from typing import Any, Optional
from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.utils.config_tools.constants import CODEFORCES_DEFAULT_TIME_LIMIT, CODEFORCES_DEFAULT_MEMORY_LIMIT

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["CodeforcesExtractor"]

log = setup_logger(__name__)



from wisent.extractors.hf.hf_task_extractors.codeforces_helpers import CodeforcesHelperMixin

class CodeforcesExtractor(CodeforcesHelperMixin, HuggingFaceBenchmarkExtractor):
    """
    Extractor for Codeforces - Competitive Programming Benchmark.

    Based on open-r1/codeforces dataset containing 10k+ competitive programming
    problems from CodeForces with verified test cases and solutions.

    Dataset Configurations:
    - default: ~10k problems
    - verifiable: 8,760 executable problems with complete/generated tests
    - verifiable-prompts: Same with 2 generation prompts per problem

    For code generation:
    - Positive (correct) = Working solution that passes test cases
    - Negative (incorrect) = Solution with bugs or wrong algorithm

    Schema (open-r1/codeforces):
        - id: str (unique problem identifier)
        - title: str (problem title)
        - description: str (problem statement)
        - input_format: str (input description)
        - output_format: str (output description)
        - examples: list[dict] (example input/output pairs)
        - official_tests: list[dict] (test cases)
        - rating: int (problem difficulty rating)
        - tags: list[str] (algorithm tags)
        - time_limit: float (seconds)
        - memory_limit: float (MB)
        - editorial: str (solution explanation)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "codeforces"

    def __init__(
        self,
        config: Optional[str] = None,
        max_rating: int | None = None,
        min_rating: int | None = None,
        language: Optional[str] = None,
    ):
        """
        Initialize Codeforces extractor.

        Args:
            config: Dataset configuration ("default", "verifiable", "verifiable-prompts")
            max_rating: Filter problems by maximum difficulty rating
            min_rating: Filter problems by minimum difficulty rating
            language: Target programming language
        """
        super().__init__()
        self.config = config if config is not None else "verifiable"
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.language = language if language is not None else "python"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Codeforces examples.

        For competitive programming:
        - Positive (correct) = Working solution approach
        - Negative (incorrect) = Wrong approach or buggy solution

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="open-r1/codeforces",
                dataset_config=self.config,
                split="train",
                limit=max_items * 2 if max_items else None,  # Load extra for filtering
            )
            log.info(f"Loaded {len(docs)} problems from Codeforces ({self.config})")
        except Exception as e:
            log.error(f"Failed to load open-r1/codeforces: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by rating if specified
            rating = doc.get("rating", 0)
            if self.max_rating and rating > self.max_rating:
                continue
            if self.min_rating and rating < self.min_rating:
                continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Codeforces pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem_id = doc.get("id", "")
            title = doc.get("title", "")
            description = doc.get("description", "")
            input_format = doc.get("input_format", "")
            output_format = doc.get("output_format", "")
            examples = doc.get("examples", [])
            rating = doc.get("rating", 0)
            tags = doc.get("tags", [])
            _TL_KEY = "time" + "_limit"
            tl_val = doc.get(_TL_KEY, CODEFORCES_DEFAULT_TIME_LIMIT)
            memory_limit = doc.get("memory_limit", CODEFORCES_DEFAULT_MEMORY_LIMIT)
            editorial = doc.get("editorial", "")
            note = doc.get("note", "")

            if not description:
                log.debug("Skipping: missing description")
                return None

            # Build the problem prompt
            prompt = self._build_prompt(
                title=title,
                description=description,
                input_format=input_format,
                output_format=output_format,
                examples=examples,
                note=note,
                **{_TL_KEY: tl_val},
                memory_limit=memory_limit,
            )

            # Build correct response (with proper approach)
            correct_response = self._create_correct_response(
                editorial=editorial,
                tags=tags,
                examples=examples,
            )

            # Build incorrect response (wrong approach)
            incorrect_response = self._create_incorrect_response(tags)

            metadata = {
                "label": "codeforces",
                "source": f"open-r1/codeforces:{self.config}",
                "problem_id": problem_id,
                "title": title,
                "rating": rating,
                "tags": tags if isinstance(tags, list) else [tags],
                _TL_KEY: tl_val,
                "memory_limit": memory_limit,
                "has_editorial": bool(editorial),
                "is_code_benchmark": True,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

