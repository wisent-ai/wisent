from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["HealthBenchExtractor"]

log = setup_logger(__name__)


class HealthBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HealthBench - OpenAI's medical/health domain benchmark.

    HealthBench evaluates LLMs on healthcare-related conversations, measuring:
    - Accuracy
    - Completeness
    - Context awareness
    - Communication quality
    - Instruction following

    The benchmark contains 5,000 multi-turn conversations developed with 262 physicians
    from 60 countries, covering 7 themes including emergency referrals, expertise-tailored
    communication, responding under uncertainty, etc.

    Schema (OnDeviceMedNotes/healthbench):
        - prompt: List[dict] (conversation with 'content' and 'role')
        - prompt_id: str (unique identifier)
        - ideal_completions_data: dict (contains ideal_completion, ideal_completions_group)
        - rubrics: List[dict] (evaluation criteria with criterion, points, tags)
        - example_tags: List[str] (categorical labels)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HealthBench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="OnDeviceMedNotes/healthbench",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} HealthBench examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid HealthBench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            prompt_data = doc.get("prompt", [])
            ideal_data = doc.get("ideal_completions_data", {})
            example_tags = doc.get("example_tags", [])
            rubrics = doc.get("rubrics", [])

            if not prompt_data:
                log.debug("Skipping: missing prompt")
                return None

            # Extract the conversation prompt
            conversation = self._format_conversation(prompt_data)
            if not conversation:
                log.debug("Skipping: could not format conversation")
                return None

            # Get the ideal completion (correct answer)
            ideal_completion = ideal_data.get("ideal_completion", "")
            if not ideal_completion:
                log.debug("Skipping: no ideal completion")
                return None

            # Create incorrect answer (a response that would fail the rubrics)
            incorrect_answer = self._create_incorrect_answer(ideal_completion, rubrics)

            # Extract the last user message as the main question
            user_messages = [m for m in prompt_data if m.get("role") == "user"]
            last_user_message = user_messages[-1].get("content", "") if user_messages else ""

            metadata = {
                "label": "healthbench",
                "source": "OnDeviceMedNotes/healthbench",
                "tags": example_tags if isinstance(example_tags, list) else [],
                "prompt_id": doc.get("prompt_id", ""),
            }

            return self._build_pair(
                question=conversation,
                correct=ideal_completion,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _format_conversation(self, prompt_data: list) -> str:
        """Format the conversation history into a prompt string."""
        if not prompt_data:
            return ""

        formatted_parts = []
        for turn in prompt_data:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                if role == "user":
                    formatted_parts.append(f"User: {content}")
                elif role == "assistant":
                    formatted_parts.append(f"Assistant: {content}")
                elif role == "system":
                    formatted_parts.append(f"System: {content}")

        if not formatted_parts:
            return ""

        conversation = "\n\n".join(formatted_parts)
        return f"{conversation}\n\nAssistant:"

    def _create_incorrect_answer(self, correct: str, rubrics: list) -> str:
        """Create an incorrect answer that would fail the evaluation rubrics."""
        # Common medical response failures:
        failure_patterns = [
            # Failure to recommend appropriate action
            "I cannot provide medical advice. Please consult a healthcare professional.",
            # Overly generic response
            "You should see a doctor about this concern.",
            # Missing important safety information
            f"Based on what you described, {correct}",
        ]

        # Check rubrics for specific failure modes
        if rubrics:
            # If there are emergency-related rubrics, create a response that misses urgency
            for rubric in rubrics:
                criterion = rubric.get("criterion", "").lower()
                if "emergency" in criterion or "urgent" in criterion:
                    return "This seems like a routine concern that can wait for your next scheduled appointment."
                if "accuracy" in criterion:
                    return f"I'm not entirely sure, but I think {correct}... However, I may be wrong about this."
                if "complete" in criterion:
                    # Return incomplete response
                    return "I'm not sure about the full details..."

        # Default: return a vague, unhelpful response
        return failure_patterns[0]

