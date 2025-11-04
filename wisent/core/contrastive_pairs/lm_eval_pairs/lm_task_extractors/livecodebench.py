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


__all__ = ["LivecodebenchExtractor"]
_LOG = setup_logger(__name__)


class LivecodebenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the LiveCodeBench benchmark using AI model submissions from HuggingFace Space."""

    LIVECODEBENCH_SPACE = "livecodebench/code_generation_samples"
    DEFAULT_MODEL = "GPT-4O-2024-08-06"  # One of the 22 available models

    def __init__(self, model_name: str | None = None):
        """
        Initialize the LiveCodeBench extractor.

        Args:
            model_name: Name of the model to extract submissions from.
                       If None, uses DEFAULT_MODEL.
                       Available models include: DeepSeek-V3, GPT-4O-2024-08-06, O1-2024-12-17, etc.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._solution_data = None
        self._problems_data = None

    def _download_space_files(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Download all_outputs.json and problems.json from HuggingFace Space.

        Returns:
            Tuple of (all_outputs dict, problems list)
        """
        try:
            from huggingface_hub import hf_hub_download

            _LOG.info(f"Downloading data from HuggingFace Space: {self.LIVECODEBENCH_SPACE}")

            # Download all_outputs.json (model submissions)
            all_outputs_path = hf_hub_download(
                repo_id=self.LIVECODEBENCH_SPACE,
                filename="all_outputs.json",
                repo_type="space"
            )

            # Download problems.json (problem metadata)
            problems_path = hf_hub_download(
                repo_id=self.LIVECODEBENCH_SPACE,
                filename="problems.json",
                repo_type="space"
            )

            with open(all_outputs_path, 'r') as f:
                all_outputs = json.load(f)

            with open(problems_path, 'r') as f:
                problems = json.load(f)

            return all_outputs, problems

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download from HuggingFace. "
                "Install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download from Space: {e}")


    def _load_solution_data(self) -> dict[str, Any]:
        """
        Load AI model solutions and create contrastive pairs.

        Downloads all_outputs.json and problems.json from HuggingFace Space
        and extracts passing/failing solutions for the specified model.

        Returns:
            Dictionary with question_id -> {good_example, bad_example} mapping.
        """
        if self._solution_data is not None:
            return self._solution_data

        # Download Space files
        all_outputs, problems = self._download_space_files()

        # Check if requested model exists
        if self.model_name not in all_outputs:
            available_models = list(all_outputs.keys())
            raise ValueError(
                f"Model '{self.model_name}' not found in all_outputs.json. "
                f"Available models: {available_models}"
            )

        # Get submissions for this model
        model_submissions = all_outputs[self.model_name]

        _LOG.info(f"Processing {len(model_submissions)} problems for model: {self.model_name}")

        # Process submissions to create solution pairs
        solution_map = self._process_submissions(model_submissions, problems)

        _LOG.info(f"Loaded {len(solution_map)} LiveCodeBench problems with solutions")

        self._solution_data = solution_map
        self._problems_data = problems
        return solution_map

    def _process_submissions(
        self,
        model_submissions: list[dict[str, Any]],
        problems: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Process model submissions to create good/bad solution pairs.

        Args:
            model_submissions: List of submissions for a specific model
                Each has: code_list, pass1_list, metadata_list
            problems: List of problem metadata from problems.json
                Each has: question_id, question_title, question_content, difficulty, etc.

        Returns:
            Dictionary with question_id -> {good_example, bad_example, difficulty}
        """
        import random

        solution_map = {}

        for problem_idx, submission in enumerate(model_submissions):
            # Get problem metadata
            if problem_idx >= len(problems):
                continue

            problem_meta = problems[problem_idx]
            question_id = problem_meta.get("question_id", str(problem_idx))

            code_list = submission.get("code_list", [])
            pass1_list = submission.get("pass1_list", [])

            # Separate passing and failing submissions
            passing_codes = []
            failing_codes = []

            for code, passed in zip(code_list, pass1_list):
                if passed:
                    passing_codes.append(code)
                else:
                    failing_codes.append(code)

            # Skip if we don't have both passing and failing examples
            if not passing_codes or not failing_codes:
                continue

            # Randomly select one passing and one failing submission
            good_code = random.choice(passing_codes)
            bad_code = random.choice(failing_codes)

            solution_map[question_id] = {
                "good_example": {"code": good_code, "passed": True},
                "bad_example": {"code": bad_code, "passed": False},
                "difficulty": problem_meta.get("difficulty", "unknown"),
                "problem_idx": problem_idx,
            }

        return solution_map

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from LiveCodeBench using actual AI model solutions from HuggingFace Space.

        Downloads submissions from livecodebench/code_generation_samples Space which contains:
        - all_outputs.json: 22 models × 880 problems with code submissions and pass/fail results
        - problems.json: Problem metadata (question_id, title, content, difficulty, etc.)

        Available models include:
            DeepSeek-V3, GPT-4O-2024-08-06, O1-2024-12-17, DeepSeek-R1-Preview,
            GPT-4-Turbo, O1-Preview, O1-Mini, and 15 more

        This method:
            1. Downloads all_outputs.json and problems.json from HuggingFace Space
            2. Extracts submissions for the specified model (default: GPT-4O-2024-08-06)
            3. Processes solutions to identify passing (good) and failing (bad) submissions
            4. Randomly selects one good and one bad example per problem
            5. Matches solutions with lm-eval task docs by question_id
            6. Creates ContrastivePair objects with actual code from the model

        Args:
            lm_eval_task_data: lm-eval task instance for LiveCodeBench.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.

        Raises:
            ValueError: If specified model not found in all_outputs.json
            RuntimeError: If download from HuggingFace Space fails
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

        for doc_idx, doc in enumerate(docs):
            pair = self._extract_pair_from_doc(doc, solution_map, doc_idx=doc_idx)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning(
                f"No valid LiveCodeBench pairs extracted from {len(docs)} docs. "
                f"Make sure model outputs are available and contain both passing and failing solutions. "
                f"Check that question_ids or problem indices match between lm-eval docs and model outputs.",
                extra={"task": task_name, "solution_count": len(solution_map)}
            )

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        solution_map: dict[str, Any],
        doc_idx: int = None
    ) -> ContrastivePair | None:
        """
        Convert a single LiveCodeBench doc into a ContrastivePair using pre-generated solutions.

        Args:
            doc: Document from lm-eval task
            solution_map: Mapping from question_id to good/bad examples
            doc_idx: Document index (used as fallback if question_id lookup fails)

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
            # Try direct question_id lookup first
            solutions = solution_map.get(question_id)

            # If not found, try using doc_idx as fallback
            if solutions is None and doc_idx is not None:
                # Try to find by problem_idx
                solutions = solution_map.get(str(doc_idx))

            if solutions is None:
                log.debug(
                    f"No pre-generated solutions found for question_id: {question_id} (doc_idx: {doc_idx})",
                    extra={"question_id": question_id, "doc_idx": doc_idx},
                )
                return None

            good_code = solutions["good_example"]["code"]
            bad_code = solutions["bad_example"]["code"]

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
                "model": self.model_name,
                "problem_idx": solutions.get("problem_idx"),
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
        # Extract model name from metadata
        model_name = metadata.get("model") if metadata else None

        # Store model name in response labels for traceability
        positive_response = PositiveResponse(
            model_response=correct_code,
            label=model_name
        )
        negative_response = NegativeResponse(
            model_response=incorrect_code,
            label=model_name
        )

        return ContrastivePair(
            prompt=prompt,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            trait_description=f"Model: {model_name}" if model_name else None,
        )
