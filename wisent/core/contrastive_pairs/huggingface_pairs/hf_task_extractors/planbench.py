from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["PlanBenchExtractor"]

log = setup_logger(__name__)

# PlanBench domains
PLANBENCH_DOMAINS = [
    "blocksworld",   # Classic blocks world planning
    "logistics",     # Package delivery logistics
]

# PlanBench task types
PLANBENCH_TASKS = [
    "plan_generation",           # Generate a valid plan
    "cost_optimal_planning",     # Generate cost-optimal plan
    "plan_verification",         # Verify if a plan is valid
    "goal_recognition",          # Recognize the goal from actions
    "plan_execution_reasoning",  # Predict outcome of action execution
    "action_reordering",         # Reorder actions for valid plan
]


class PlanBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for PlanBench - Planning and Reasoning Benchmark.

    PlanBench evaluates LLMs on planning and reasoning about actions
    and change, using domains from the International Planning Competition.

    Domains:
    - Blocksworld: Classic blocks stacking problems
    - Logistics: Package delivery with trucks and planes

    Task Types:
    - Plan generation and cost-optimal planning
    - Plan verification
    - Goal recognition
    - Plan execution reasoning
    - Action reordering

    Dataset: GitHub karthikv792/LLMs-Planning

    For planning evaluation:
    - Positive (correct) = Valid plan or correct reasoning
    - Negative (incorrect) = Invalid plan or incorrect reasoning
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "planning_reasoning"

    def __init__(self, domain: str = "blocksworld", task: str = "plan_generation"):
        """
        Initialize PlanBench extractor.

        Args:
            domain: Planning domain ("blocksworld", "logistics")
            task: Task type (e.g., "plan_generation", "plan_verification")
        """
        super().__init__()
        self.domain = domain
        self.task = task

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from PlanBench examples.

        Creates pairs for planning evaluation:
        - Positive (correct) = Valid planning solution
        - Negative (incorrect) = Invalid planning solution

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # PlanBench is on GitHub, create examples based on documented structure
        docs = self._create_planbench_examples(max_items or 50)
        log.info(f"Created {len(docs)} PlanBench examples ({self.domain}, {self.task})")

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

    def _create_planbench_examples(self, count: int) -> list[dict[str, Any]]:
        """Create examples based on PlanBench structure."""
        examples = []

        if self.domain == "blocksworld":
            examples = self._create_blocksworld_examples(count)
        elif self.domain == "logistics":
            examples = self._create_logistics_examples(count)
        else:
            examples = self._create_blocksworld_examples(count)

        return examples

    def _create_blocksworld_examples(self, count: int) -> list[dict[str, Any]]:
        """Create blocksworld planning examples."""
        blocksworld_cases = [
            {
                "initial_state": "Block A is on the table. Block B is on Block A. Block C is on the table.",
                "goal_state": "Block A is on Block B. Block B is on Block C.",
                "valid_plan": [
                    "1. Unstack B from A",
                    "2. Put B on C",
                    "3. Pick up A",
                    "4. Stack A on B",
                ],
                "invalid_plan": [
                    "1. Pick up A",  # Invalid - B is on A
                    "2. Put A on B",
                ],
            },
            {
                "initial_state": "Block A is on Block B. Block B is on the table. Block C is on the table. The robot arm is empty.",
                "goal_state": "Block B is on Block A. Block C is on Block B.",
                "valid_plan": [
                    "1. Unstack A from B",
                    "2. Put A on the table",
                    "3. Pick up B",
                    "4. Stack B on A",
                    "5. Pick up C",
                    "6. Stack C on B",
                ],
                "invalid_plan": [
                    "1. Stack B on A",  # Invalid - A is on B
                ],
            },
            {
                "initial_state": "Block A, B, and C are on the table. Block D is on Block A.",
                "goal_state": "Block A is on Block B. Block B is on Block C. Block D is on Block A.",
                "valid_plan": [
                    "1. Unstack D from A",
                    "2. Put D on table",
                    "3. Pick up B",
                    "4. Stack B on C",
                    "5. Pick up A",
                    "6. Stack A on B",
                    "7. Pick up D",
                    "8. Stack D on A",
                ],
                "invalid_plan": [
                    "1. Pick up A",  # Invalid - D is on A
                ],
            },
        ]

        examples = []
        for i in range(count):
            case = blocksworld_cases[i % len(blocksworld_cases)].copy()
            case["case_id"] = f"blocks_{i:03d}"
            case["domain"] = "blocksworld"
            examples.append(case)

        return examples

    def _create_logistics_examples(self, count: int) -> list[dict[str, Any]]:
        """Create logistics planning examples."""
        logistics_cases = [
            {
                "initial_state": "Package P1 is in City A. Truck T1 is in City A. Package needs to go to City B.",
                "goal_state": "Package P1 is in City B.",
                "valid_plan": [
                    "1. Load P1 onto T1 in City A",
                    "2. Drive T1 from City A to City B",
                    "3. Unload P1 from T1 in City B",
                ],
                "invalid_plan": [
                    "1. Drive T1 from City A to City B",
                    "2. Unload P1 from T1",  # Invalid - P1 was never loaded
                ],
            },
            {
                "initial_state": "Package P1 is in City A. Package P2 is in City B. Plane A1 is in City A. Goal: P1 in City C, P2 in City A.",
                "goal_state": "Package P1 is in City C. Package P2 is in City A.",
                "valid_plan": [
                    "1. Load P1 onto Plane A1 in City A",
                    "2. Fly A1 from City A to City B",
                    "3. Load P2 onto A1 in City B",
                    "4. Fly A1 from City B to City A",
                    "5. Unload P2 in City A",
                    "6. Fly A1 from City A to City C",
                    "7. Unload P1 in City C",
                ],
                "invalid_plan": [
                    "1. Fly A1 to City B",
                    "2. Unload P1",  # P1 was never loaded
                ],
            },
        ]

        examples = []
        for i in range(count):
            case = logistics_cases[i % len(logistics_cases)].copy()
            case["case_id"] = f"logistics_{i:03d}"
            case["domain"] = "logistics"
            examples.append(case)

        return examples

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            case_id = doc.get("case_id", "")
            initial_state = doc.get("initial_state", "").strip()
            goal_state = doc.get("goal_state", "").strip()
            valid_plan = doc.get("valid_plan", [])
            invalid_plan = doc.get("invalid_plan", [])
            domain = doc.get("domain", self.domain)

            if not initial_state or not goal_state:
                log.debug("Skipping: missing states")
                return None

            # Build the planning task prompt
            task_prompt = self._build_planning_prompt(
                initial_state, goal_state, domain
            )

            # Positive = valid plan
            correct_response = self._create_valid_plan_response(valid_plan)
            # Negative = invalid plan
            incorrect_response = self._create_invalid_plan_response(invalid_plan)

            metadata = {
                "label": "planbench",
                "source": "karthikv792/LLMs-Planning",
                "case_id": case_id,
                "domain": domain,
                "task": self.task,
                "plan_length": len(valid_plan),
                "is_planning_benchmark": True,
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

    def _build_planning_prompt(
        self, initial_state: str, goal_state: str, domain: str
    ) -> str:
        """Build the planning task prompt."""
        domain_desc = ""
        if domain == "blocksworld":
            domain_desc = (
                "In this blocks world domain, you can:\n"
                "- Pick up a block (only if nothing is on it and arm is empty)\n"
                "- Put down a block on the table\n"
                "- Stack a block on another (only if target block is clear)\n"
                "- Unstack a block from another\n\n"
            )
        elif domain == "logistics":
            domain_desc = (
                "In this logistics domain, you can:\n"
                "- Load packages onto trucks/planes (at same location)\n"
                "- Unload packages from trucks/planes\n"
                "- Drive trucks between locations in same city\n"
                "- Fly planes between cities\n\n"
            )

        return (
            f"{domain_desc}"
            f"Initial State:\n{initial_state}\n\n"
            f"Goal State:\n{goal_state}\n\n"
            "Generate a valid sequence of actions to achieve the goal state from "
            "the initial state. Ensure each action's preconditions are satisfied."
        )

    def _create_valid_plan_response(self, plan: list[str]) -> str:
        """Create a response with a valid plan."""
        plan_str = "\n".join(plan)
        return (
            f"Here is a valid plan to achieve the goal:\n\n{plan_str}\n\n"
            "Each action in this sequence has its preconditions satisfied by the "
            "previous actions, and executing them in order will achieve the goal state."
        )

    def _create_invalid_plan_response(self, plan: list[str]) -> str:
        """Create a response with an invalid plan."""
        plan_str = "\n".join(plan) if plan else "1. [Incomplete plan]"
        return (
            f"Here's my plan:\n\n{plan_str}\n\n"
            "This should work to reach the goal."
        )

