from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["TauBenchExtractor"]

log = setup_logger(__name__)

# TAU-bench domains
TAU_BENCH_DOMAINS = [
    "retail",      # E-commerce customer service
    "airline",     # Airline booking and support
    "telecom",     # Telecommunications support (tau2-bench)
]


class TauBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for τ-bench - Tool-Agent-User Interaction Benchmark (Sierra 2024).

    τ-bench evaluates language agents on completing complex tasks while
    interacting with simulated users and tools in real-world domains.

    Domains:
    - retail: E-commerce customer service scenarios
    - airline: Airline booking and support scenarios
    - telecom: Telecommunications support (τ²-bench extension)

    Dataset: HuggingFaceH4/tau2-bench-data (or GitHub sierra-research/tau-bench)

    Schema:
        - id: str (task identifier)
        - user_scenario: str (user situation description)
        - description: str (task description)
        - evaluation_criteria: list (success criteria)
        - initial_state: dict (starting database state)

    For agent task evaluation:
    - Positive (correct) = Successfully completes user task
    - Negative (incorrect) = Fails to complete or makes errors
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "agent_task_completion"

    def __init__(self, domain: str = "retail"):
        """
        Initialize TAU-bench extractor.

        Args:
            domain: Domain to use ("retail", "airline", "telecom")
        """
        super().__init__()
        self.domain = domain

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TAU-bench examples.

        Creates pairs for agent task completion:
        - Positive (correct) = Successfully completes the task
        - Negative (incorrect) = Fails or makes errors

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try to load from HuggingFace
        try:
            docs = self.load_dataset(
                dataset_name="HuggingFaceH4/tau2-bench-data",
                split="train",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from tau2-bench-data")
        except Exception as e:
            log.warning(f"Failed to load tau2-bench from HF: {e}")
            # Create examples based on TAU-bench structure
            docs = self._create_synthetic_examples(max_items or 50)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid TAU-bench pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic examples based on TAU-bench structure."""
        examples = []

        # Retail domain examples
        retail_examples = [
            {
                "id": "retail_001",
                "domain": "retail",
                "user_scenario": "Customer wants to return an item purchased last week due to wrong size",
                "description": "Process a return request for order #12345, item: Blue T-Shirt (Size M), verify return eligibility, initiate return process",
                "evaluation_criteria": [
                    "Verify order exists",
                    "Check return window (30 days)",
                    "Initiate return label",
                    "Update order status",
                ],
                "available_tools": [
                    "get_order_details",
                    "check_return_eligibility",
                    "create_return_label",
                    "update_order_status",
                ],
            },
            {
                "id": "retail_002",
                "domain": "retail",
                "user_scenario": "Customer wants to track their package and update delivery address",
                "description": "Look up tracking for order #67890, update delivery address to new location if package hasn't shipped",
                "evaluation_criteria": [
                    "Retrieve tracking information",
                    "Check shipment status",
                    "Update address if allowed",
                    "Confirm changes with customer",
                ],
                "available_tools": [
                    "get_tracking_info",
                    "check_shipment_status",
                    "update_delivery_address",
                    "send_confirmation",
                ],
            },
        ]

        # Airline domain examples
        airline_examples = [
            {
                "id": "airline_001",
                "domain": "airline",
                "user_scenario": "Passenger needs to change flight from tomorrow to next week due to emergency",
                "description": "Modify booking ABC123, change departure date, check fare difference, process change fee if applicable",
                "evaluation_criteria": [
                    "Retrieve booking details",
                    "Check availability on new date",
                    "Calculate fare difference",
                    "Process modification",
                ],
                "available_tools": [
                    "get_booking",
                    "search_flights",
                    "calculate_fare_difference",
                    "modify_booking",
                ],
            },
            {
                "id": "airline_002",
                "domain": "airline",
                "user_scenario": "Customer requesting seat change and meal preference update for upcoming flight",
                "description": "Update seat assignment to window seat and add vegetarian meal for booking XYZ789",
                "evaluation_criteria": [
                    "Verify booking exists",
                    "Check seat availability",
                    "Update seat assignment",
                    "Add meal preference",
                ],
                "available_tools": [
                    "get_booking",
                    "get_seat_map",
                    "assign_seat",
                    "update_meal_preference",
                ],
            },
        ]

        # Alternate between domains
        all_examples = []
        if self.domain == "retail":
            all_examples = retail_examples
        elif self.domain == "airline":
            all_examples = airline_examples
        else:
            all_examples = retail_examples + airline_examples

        for i in range(count):
            example = all_examples[i % len(all_examples)].copy()
            example["id"] = f"{example['domain']}_{i:03d}"
            examples.append(example)

        return examples

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            task_id = doc.get("id", "")
            user_scenario = doc.get("user_scenario", "").strip()
            description = doc.get("description", "").strip()
            evaluation_criteria = doc.get("evaluation_criteria", [])
            available_tools = doc.get("available_tools", [])
            domain = doc.get("domain", self.domain)

            if not user_scenario and not description:
                log.debug("Skipping: missing scenario or description")
                return None

            # Build the agent task prompt
            task_prompt = self._build_task_prompt(
                user_scenario, description, available_tools
            )

            # Positive = successful task completion
            correct_response = self._create_successful_response(
                description, evaluation_criteria, available_tools
            )
            # Negative = failed or incomplete task
            incorrect_response = self._create_failed_response(description)

            metadata = {
                "label": "tau_bench",
                "source": "sierra-research/tau-bench",
                "task_id": task_id,
                "domain": domain,
                "num_criteria": len(evaluation_criteria) if evaluation_criteria else 0,
                "num_tools": len(available_tools) if available_tools else 0,
                "is_agent_benchmark": True,
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

    def _build_task_prompt(
        self,
        user_scenario: str,
        description: str,
        available_tools: list[str],
    ) -> str:
        """Build the agent task prompt."""
        parts = [f"User Scenario: {user_scenario}"]

        if description:
            parts.append(f"\nTask: {description}")

        if available_tools:
            tools_str = ", ".join(available_tools)
            parts.append(f"\nAvailable Tools: {tools_str}")

        parts.append("\nPlease help the user complete their request using the available tools.")

        return "\n".join(parts)

    def _create_successful_response(
        self,
        description: str,
        criteria: list[str],
        tools: list[str],
    ) -> str:
        """Create a successful task completion response."""
        steps = []
        for i, tool in enumerate(tools):
            criterion = criteria[i] if i < len(criteria) else f"Execute {tool}"
            steps.append(f"{i+1}. {criterion} using {tool}")

        steps_str = "\n".join(steps) if steps else "Complete the requested actions"

        return (
            f"I'll help you with this request. Let me work through the necessary steps:\n\n"
            f"{steps_str}\n\n"
            "I have successfully completed all the required actions. The task has been "
            "processed and the changes have been applied to your account. Is there "
            "anything else I can help you with?"
        )

    def _create_failed_response(self, description: str) -> str:
        """Create a failed task response."""
        return (
            "I apologize, but I'm having trouble completing this request. "
            "I attempted to process your request but encountered an issue. "
            "The system isn't responding as expected and I couldn't complete "
            "all the necessary steps. Please try again later or contact support."
        )

