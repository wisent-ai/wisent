from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AlpacaEvalExtractor"]

log = setup_logger(__name__)


class AlpacaEvalExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for AlpacaEval 2.0 - Instruction-following evaluation benchmark.

    AlpacaEval is an automatic evaluator for instruction-following language models.
    It measures the win rate against a reference model (GPT-4 Turbo for 2.0).

    The benchmark contains 805 instructions from 5 distinct test sets.

    Schema (tatsu-lab/alpaca_eval):
        - instruction: str (the instruction/prompt)
        - output: str (model's response)
        - generator: str (model that generated the output)
        - dataset: str (original source test set)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from AlpacaEval examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        # AlpacaEval has a custom loading script, so we use trust_remote_code
        try:
            docs = self.load_dataset(
                dataset_name="tatsu-lab/alpaca_eval",
                split="eval",
                limit=max_items,
                trust_remote_code=True,
            )
        except Exception as e:
            log.warning(f"Failed to load with trust_remote_code, trying alternative: {e}")
            # Try loading without the custom script
            docs = self.load_dataset(
                dataset_name="tatsu-lab/alpaca_eval",
                split="eval",
                limit=max_items,
            )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} AlpacaEval examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid AlpacaEval pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            instruction = doc.get("instruction", "").strip()
            output = doc.get("output", "").strip()
            generator = doc.get("generator", "")
            dataset = doc.get("dataset", "")

            if not instruction:
                log.debug("Skipping: missing instruction")
                return None

            # Use the reference output as the correct answer (if available)
            # Otherwise use the provided output
            correct_answer = output if output else "I'd be happy to help with that."

            # Create an incorrect answer (a poor quality response)
            incorrect_answer = self._create_incorrect_answer(instruction)

            # Format the prompt
            formatted_question = f"{instruction}"

            metadata = {
                "label": "alpaca_eval",
                "source": "tatsu-lab/alpaca_eval",
                "generator": generator,
                "original_dataset": dataset,
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, instruction: str) -> str:
        """Create a low-quality response that would lose to a good model."""
        # Common low-quality response patterns:
        low_quality_patterns = [
            # Too short/unhelpful
            "I don't know.",
            # Refuses unnecessarily
            "I cannot help with that request.",
            # Off-topic
            "That's an interesting question. Let me tell you about something else entirely.",
            # Repetitive/filler
            "Sure, I can help. Let me think about that. Actually, I need more information to provide a good answer.",
        ]

        # Select based on instruction characteristics
        instruction_lower = instruction.lower()

        if "write" in instruction_lower or "create" in instruction_lower:
            return "I'm not sure how to write that. Can you be more specific?"
        elif "explain" in instruction_lower or "what is" in instruction_lower:
            return "I don't have enough information to explain that properly."
        elif "code" in instruction_lower or "program" in instruction_lower:
            return "# TODO: implement this\npass"
        elif "list" in instruction_lower:
            return "Here are some items:\n1. Item\n2. Another item\n3. More items"
        else:
            return low_quality_patterns[0]

