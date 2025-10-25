from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["LiveCodeBenchExtractor"]
_LOG = setup_logger(__name__)


class LiveCodeBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the LiveCodeBench benchmark using pre-generated AI model solutions."""

    def __init__(self, wisent_core_path: str | None = None):
        """
        Initialize the LiveCodeBench extractor.

        Args:
            wisent_core_path: Path to wisent-core repository containing question_examples.json.
                             If None, will attempt to find it automatically.
        """
        self.wisent_core_path = wisent_core_path
        self._solution_data = None

    def _find_wisent_core_path(self) -> Path | None:
        """
        Attempt to automatically find the wisent-core repository path.

        Returns:
            Path to wisent-core if found, None otherwise.
        """
        # Try common locations relative to current file
        current_file = Path(__file__).resolve()

        # Go up to backends directory and look for sibling wisent-core
        backends_dir = current_file.parents[6]  # Go up from lm_task_extractors to backends/
        wisent_dir = backends_dir.parent  # Go to Wisent/

        possible_paths = [
            wisent_dir / "wisent-core" / "wisent-core",
            wisent_dir / "wisent-core",
            Path.home() / "Documents" / "CodingProjects" / "Wisent" / "wisent-core" / "wisent-core",
        ]

        for path in possible_paths:
            if path.exists() and (path / "question_examples.json").exists():
                return path

        return None

    def _load_solution_data(self) -> dict[str, Any]:
        """
        Load pre-generated AI model solutions from wisent-core.

        Returns:
            Dictionary with question_id -> {good_example, bad_example} mapping.

        Raises:
            FileNotFoundError: If question_examples.json cannot be found.
        """
        if self._solution_data is not None:
            return self._solution_data

        # Find wisent-core path
        if self.wisent_core_path:
            base_path = Path(self.wisent_core_path)
        else:
            base_path = self._find_wisent_core_path()
            if base_path is None:
                raise FileNotFoundError(
                    "Could not find wisent-core repository. Please specify wisent_core_path "
                    "when initializing LiveCodeBenchExtractor, or ensure wisent-core is in "
                    "the expected location."
                )

        question_file = base_path / "question_examples.json"

        if not question_file.exists():
            raise FileNotFoundError(
                f"question_examples.json not found at {question_file}. "
                f"Please ensure wisent-core repository is properly set up."
            )

        _LOG.info(f"Loading LiveCodeBench solutions from {question_file}")

        with open(question_file, 'r') as f:
            data = json.load(f)

        # Create mapping from question_id to solutions
        solution_map = {}
        for question in data.get("questions", []):
            question_id = question.get("question_id")
            if question_id and question.get("good_example") and question.get("bad_example"):
                solution_map[question_id] = {
                    "good_example": question["good_example"],
                    "bad_example": question["bad_example"],
                    "difficulty": question.get("difficulty", "unknown"),
                }

        _LOG.info(f"Loaded {len(solution_map)} LiveCodeBench problems with solutions")

        self._solution_data = solution_map
        return solution_map

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from LiveCodeBench docs using pre-generated AI solutions.

        LiveCodeBench schema (from lm-eval task):
            - question_id: str
            - question_title: str
            - question_content: str
            - starter_code: str (optional)
            - difficulty: str
            - platform: str

        Solutions are loaded from wisent-core's question_examples.json which contains:
            - good_example: {model: str, code: str, result: "good"}
            - bad_example: {model: str, code: str, result: "bad"}

        Args:
            lm_eval_task_data: lm-eval task instance for LiveCodeBench.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "livecodebench"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        # Load pre-generated solutions
        try:
            solution_map = self._load_solution_data()
        except FileNotFoundError as e:
            log.error(f"Failed to load solution data: {e}")
            return []

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, solution_map)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning(
                f"No valid LiveCodeBench pairs extracted from {len(docs)} docs. "
                f"Make sure question_ids in lm-eval match those in wisent-core.",
                extra={"task": task_name}
            )

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        solution_map: dict[str, Any]
    ) -> ContrastivePair | None:
        """
        Convert a single LiveCodeBench doc into a ContrastivePair using pre-generated solutions.

        Args:
            doc: Document from lm-eval task
            solution_map: Mapping from question_id to good/bad examples

        Returns:
            ContrastivePair if solutions exist for this problem, None otherwise.
        """
        log = bind(_LOG, doc_id=doc.get("question_id", "unknown"))

        try:
            question_id = doc.get("question_id")
            question_title = doc.get("question_title", "").strip()
            question_content = doc.get("question_content", "").strip()
            starter_code = doc.get("starter_code", "").strip()

            if not question_id or not question_content:
                log.debug(
                    "Skipping doc due to missing question_id or content",
                    extra={"doc": doc},
                )
                return None

            # Look up pre-generated solutions
            if question_id not in solution_map:
                log.debug(
                    f"No pre-generated solutions found for question_id: {question_id}",
                    extra={"question_id": question_id},
                )
                return None

            solutions = solution_map[question_id]
            good_code = solutions["good_example"]["code"]
            bad_code = solutions["bad_example"]["code"]
            good_model = solutions["good_example"].get("model", "unknown")
            bad_model = solutions["bad_example"].get("model", "unknown")

            # Build prompt from problem description
            prompt_parts = []
            if question_title:
                prompt_parts.append(f"Problem: {question_title}")
            prompt_parts.append(question_content)
            if starter_code:
                prompt_parts.append(f"\nStarter Code:\n{starter_code}")

            prompt = "\n\n".join(prompt_parts)

            metadata = {
                "label": "livecodebench",
                "question_id": question_id,
                "difficulty": solutions.get("difficulty", "unknown"),
                "good_model": good_model,
                "bad_model": bad_model,
            }

            return self._build_pair(
                prompt=prompt,
                correct_code=good_code,
                incorrect_code=bad_code,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        prompt: str,
        correct_code: str,
        incorrect_code: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """
        Build a ContrastivePair from a coding problem and correct/incorrect solutions.

        Args:
            prompt: The coding problem description
            correct_code: Code that passes all tests (positive example)
            incorrect_code: Code that fails tests (negative example)
            metadata: Additional metadata about the problem

        Returns:
            ContrastivePair object
        """
        positive_response = PositiveResponse(model_response=correct_code)
        negative_response = NegativeResponse(model_response=incorrect_code)

        return ContrastivePair(
            prompt=prompt,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
        )
