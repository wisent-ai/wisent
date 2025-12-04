from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["TravelPlannerExtractor"]

log = setup_logger(__name__)


class TravelPlannerExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for TravelPlanner - Real-World Planning Benchmark (ICML 2024 Spotlight).

    TravelPlanner evaluates language agents on complex travel planning tasks
    with multiple constraints (environment, commonsense, hard constraints).

    Contains 1,225 queries requiring agents to:
    - Gather information using tools
    - Plan transportation, meals, attractions, accommodation
    - Satisfy multiple constraint types

    Dataset: osunlp/TravelPlanner

    Schema:
        - split: str (train/validation/test)
        - org: str (origin city)
        - dest: str (destination city/cities)
        - days: int (trip duration)
        - date: str (trip start date)
        - query: str (user's travel request)
        - level: str (difficulty level)
        - reference_information: str (available data reference)
        - visiting_city_number: int (number of cities)
        - people_number: int (travelers count)
        - local_constraint: str (specific constraints)
        - budget: int (budget limit)
        - annotated_plan: str (reference plan, train only)

    For planning evaluation:
    - Positive (correct) = Valid plan satisfying all constraints
    - Negative (incorrect) = Invalid plan violating constraints
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "planning_constraint_satisfaction"

    def __init__(self, split: str = "validation"):
        """
        Initialize TravelPlanner extractor.

        Args:
            split: Dataset split ("train", "validation", "test")
        """
        super().__init__()
        self.split = split

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TravelPlanner examples.

        Creates pairs for planning task evaluation:
        - Positive (correct) = Valid plan satisfying constraints
        - Negative (incorrect) = Plan that violates constraints

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            # TravelPlanner configs: train, validation, test
            # Each config has a single split with same name as config
            docs = self.load_dataset(
                dataset_name="osunlp/TravelPlanner",
                dataset_config=self.split,
                split=self.split,  # Split name matches config name
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from TravelPlanner ({self.split})")
        except Exception as e:
            log.error(f"Failed to load TravelPlanner: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid TravelPlanner pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            org = doc.get("org", "").strip()
            dest = doc.get("dest", "").strip()
            days = doc.get("days", 1)
            date = doc.get("date", "").strip()
            query = doc.get("query", "").strip()
            level = doc.get("level", "").strip()
            people_number = doc.get("people_number", 1)
            budget = doc.get("budget", 0)
            local_constraint = doc.get("local_constraint", "")
            annotated_plan = doc.get("annotated_plan", "")

            if not query:
                log.debug("Skipping: missing query")
                return None

            # Build the planning task prompt
            task_prompt = self._build_planning_prompt(
                query=query,
                org=org,
                dest=dest,
                days=days,
                date=date,
                people=people_number,
                budget=budget,
                constraint=local_constraint,
            )

            # Positive = valid plan (use annotated if available, else generate)
            if annotated_plan:
                correct_response = self._format_annotated_plan(annotated_plan, org, dest, days)
            else:
                correct_response = self._create_valid_plan(org, dest, days, budget, local_constraint)

            # Negative = plan with constraint violations
            incorrect_response = self._create_invalid_plan(org, dest, days, budget, local_constraint)

            metadata = {
                "label": "travelplanner",
                "source": "osunlp/TravelPlanner",
                "origin": org,
                "destination": dest,
                "days": days,
                "level": level,
                "people_number": people_number,
                "budget": budget,
                "has_constraint": bool(local_constraint),
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
        self,
        query: str,
        org: str,
        dest: str,
        days: int,
        date: str,
        people: int,
        budget: int,
        constraint: str,
    ) -> str:
        """Build the travel planning task prompt."""
        parts = [f"Travel Planning Request: {query}"]

        details = []
        if org:
            details.append(f"Origin: {org}")
        if dest:
            details.append(f"Destination: {dest}")
        if days:
            details.append(f"Duration: {days} days")
        if date:
            details.append(f"Start Date: {date}")
        if people and people > 1:
            details.append(f"Travelers: {people} people")
        if budget:
            details.append(f"Budget: ${budget}")

        if details:
            parts.append("\nTrip Details:\n- " + "\n- ".join(details))

        if constraint:
            parts.append(f"\nSpecial Requirements: {constraint}")

        parts.append(
            "\nPlease create a detailed travel plan including transportation, "
            "accommodation, meals, and attractions for each day. Ensure the plan "
            "satisfies all constraints and stays within budget."
        )

        return "\n".join(parts)

    def _format_annotated_plan(
        self, annotated_plan: str, org: str, dest: str, days: int
    ) -> str:
        """Format the annotated plan as a response."""
        if isinstance(annotated_plan, str) and annotated_plan.strip():
            return f"Here's your travel plan from {org} to {dest}:\n\n{annotated_plan}"
        return self._create_valid_plan(org, dest, days, 0, "")

    def _create_valid_plan(
        self,
        org: str,
        dest: str,
        days: int,
        budget: int,
        constraint: str,
    ) -> str:
        """Create a valid plan that satisfies constraints."""
        plan_parts = [f"Here's your {days}-day travel plan from {org} to {dest}:\n"]

        for day in range(1, days + 1):
            day_plan = f"\n**Day {day}:**"
            if day == 1:
                day_plan += f"\n- Morning: Depart from {org}, arrive at {dest}"
                day_plan += "\n- Afternoon: Check into hotel, explore nearby area"
            elif day == days:
                day_plan += "\n- Morning: Final sightseeing"
                day_plan += f"\n- Afternoon: Depart from {dest}, return to {org}"
            else:
                day_plan += f"\n- Morning: Visit local attractions in {dest}"
                day_plan += "\n- Afternoon: Cultural activities and dining"
            day_plan += "\n- Evening: Dinner at local restaurant"
            plan_parts.append(day_plan)

        if budget:
            plan_parts.append(f"\n\nEstimated total cost: ${int(budget * 0.9)} (within your ${budget} budget)")

        if constraint:
            plan_parts.append(f"\n\nYour requirements have been accommodated: {constraint}")

        return "".join(plan_parts)

    def _create_invalid_plan(
        self,
        org: str,
        dest: str,
        days: int,
        budget: int,
        constraint: str,
    ) -> str:
        """Create an invalid plan that violates constraints."""
        plan_parts = [f"Here's a quick {days}-day trip plan:\n"]

        # Create a plan that violates budget and ignores constraints
        for day in range(1, days + 1):
            day_plan = f"\n**Day {day}:**"
            day_plan += "\n- Book first-class flight (most expensive option)"
            day_plan += "\n- Stay at luxury 5-star resort"
            day_plan += "\n- Private guided tours all day"
            day_plan += "\n- Fine dining at Michelin restaurants"
            plan_parts.append(day_plan)

        # Violate budget constraint
        if budget:
            inflated_cost = budget * 3
            plan_parts.append(f"\n\nEstimated total cost: ${inflated_cost} (exceeds your ${budget} budget)")

        # Ignore the constraint
        if constraint:
            plan_parts.append(f"\n\nNote: Unable to accommodate your requirement '{constraint}' with this itinerary.")

        return "".join(plan_parts)

