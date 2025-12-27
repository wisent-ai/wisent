from __future__ import annotations

from typing import Any
from datasets import load_dataset
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["PlanBenchExtractor"]

log = setup_logger(__name__)

PLANBENCH_CONFIGS = [
    "task_1_plan_generation",
    "task_2_plan_optimality",
    "task_3_plan_verification",
    "task_5_plan_generalization",
    "task_7_plan_execution",
    "task_8_1_goal_shuffling",
    "task_8_2_full_to_partial",
]


class PlanBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for PlanBench - Planning and Reasoning Benchmark (NeurIPS 2023).

    PlanBench evaluates LLMs on planning and reasoning about actions
    and change, using domains from the International Planning Competition.

    Dataset: tasksource/planbench (HuggingFace)

    Available configs:
    - task_1_plan_generation: Generate a valid plan
    - task_2_plan_optimality: Generate cost-optimal plan
    - task_3_plan_verification: Verify if a plan is valid
    - task_5_plan_generalization: Generalize plan to new instances
    - task_7_plan_execution: Predict execution outcome
    - task_8_1_goal_shuffling: Handle shuffled goals
    - task_8_2_full_to_partial: Full to partial observability

    For planning evaluation:
    - Positive (correct) = Valid plan matching ground truth
    - Negative (incorrect) = Invalid or wrong plan
    """

    evaluator_name = "planning_reasoning"

    def __init__(self, config: str = "task_1_plan_generation"):
        """
        Initialize PlanBench extractor.

        Args:
            config: PlanBench task config (default: task_1_plan_generation)
        """
        super().__init__()
        if config not in PLANBENCH_CONFIGS:
            log.warning(f"Unknown config '{config}', using task_1_plan_generation")
            config = "task_1_plan_generation"
        self.config = config

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from PlanBench.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        docs = self._load_planbench_data()
        log.info(f"Loaded {len(docs)} PlanBench examples (config: {self.config})")

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid PlanBench pairs extracted")

        return pairs

    def _load_planbench_data(self) -> list[dict[str, Any]]:
        """Load PlanBench data from HuggingFace."""
        try:
            ds = load_dataset("tasksource/planbench", self.config, split="train")
            examples = []
            for i, item in enumerate(ds):
                examples.append({
                    "case_id": f"planbench_{self.config}_{i:04d}",
                    "task": item.get("task", ""),
                    "prompt_type": item.get("prompt_type", ""),
                    "domain": item.get("domain", ""),
                    "instance_id": item.get("instance_id", ""),
                    "query": item.get("query", ""),
                    "ground_truth_plan": item.get("ground_truth_plan", ""),
                })
            return examples
        except Exception as e:
            log.error(f"Failed to load PlanBench from HuggingFace: {e}")
            raise RuntimeError(f"Cannot load PlanBench data: {e}")

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        
        PlanBench HuggingFace format:
        {"task": "task_1_plan_generation", "prompt_type": "oneshot", "domain": "...",
         "instance_id": 2, "query": "...", "ground_truth_plan": "..."}
        """
        try:
            case_id = doc.get("case_id", "")
            query = doc.get("query", "").strip()
            ground_truth_plan = doc.get("ground_truth_plan", "").strip()
            domain = doc.get("domain", "")
            task = doc.get("task", "")

            if not query or not ground_truth_plan:
                log.debug("Skipping: missing query or ground_truth_plan")
                return None

            correct_response = self._create_correct_response(ground_truth_plan)
            incorrect_response = self._create_incorrect_response(ground_truth_plan)

            metadata = {
                "label": "planbench",
                "source": "tasksource/planbench",
                "case_id": case_id,
                "domain": domain,
                "task": task,
                "config": self.config,
                "is_planning_benchmark": True,
            }

            return self._build_pair(
                question=query,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_correct_response(self, ground_truth_plan: str) -> str:
        """Create a response with the correct plan."""
        return (
            f"Here is the plan to achieve the goal:\n\n{ground_truth_plan}\n\n"
            "Each action in this sequence has its preconditions satisfied."
        )

    def _create_incorrect_response(self, ground_truth_plan: str) -> str:
        """Create an incorrect response (wrong/incomplete plan)."""
        lines = ground_truth_plan.strip().split("\n")
        if len(lines) > 1:
            wrong_plan = "\n".join(reversed(lines[:2]))
        else:
            wrong_plan = "(noop)"
        return (
            f"Here's my plan:\n\n{wrong_plan}\n\n"
            "This should work to reach the goal."
        )

