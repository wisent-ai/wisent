from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["ArenaHardExtractor"]

log = setup_logger(__name__)


class ArenaHardExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Arena-Hard-Auto - Challenging LLM evaluation benchmark from LMSYS.

    Arena-Hard-Auto contains 500 challenging user queries sourced from Chatbot Arena.
    It has the highest correlation with Chatbot Arena among popular open-ended LLM benchmarks.

    Models are evaluated by comparing responses against a baseline (GPT-4-0314) using
    GPT-4-Turbo as a judge.

    Schema (lmsys/arena-hard-auto-v0.1):
        - question_id: str (unique identifier, 32 chars)
        - category: str (question category)
        - cluster: str (cluster grouping)
        - turns: List[dict] (conversation turns with 'content')
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Arena-Hard examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="lmsys/arena-hard-auto-v0.1",
            split="train",  # Arena-Hard uses train split
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} Arena-Hard examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Arena-Hard pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question_id = doc.get("question_id", "")
            category = doc.get("category", "")
            cluster = doc.get("cluster", "")
            turns = doc.get("turns", [])

            if not turns:
                log.debug("Skipping: no turns found")
                return None

            # Extract the question from the first turn
            first_turn = turns[0] if turns else {}
            question = first_turn.get("content", "").strip()

            if not question:
                log.debug("Skipping: empty question")
                return None

            # For Arena-Hard, we don't have reference answers in the dataset
            # We create a high-quality response template and a low-quality one
            correct_answer = self._create_quality_response(question, cluster)
            incorrect_answer = self._create_poor_response(question)

            metadata = {
                "label": "arena_hard",
                "source": "lmsys/arena-hard-auto-v0.1",
                "question_id": question_id,
                "category": category,
                "cluster": cluster,
            }

            return self._build_pair(
                question=question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_quality_response(self, question: str, cluster: str) -> str:
        """Create a template for a high-quality response.

        Note: In actual evaluation, this would be replaced with model outputs.
        This template serves as a placeholder showing characteristics of good responses.
        """
        # Provide a structured, helpful response template
        cluster_lower = cluster.lower() if cluster else ""

        if "code" in cluster_lower or "programming" in cluster_lower:
            return (
                "I'll help you with that programming task. Here's a detailed solution:\n\n"
                "```python\n# Solution code here\n```\n\n"
                "This approach works because... Let me explain the key aspects..."
            )
        elif "math" in cluster_lower:
            return (
                "Let me solve this step by step:\n\n"
                "Step 1: First, we identify...\n"
                "Step 2: Next, we apply...\n"
                "Step 3: Therefore, the answer is...\n\n"
                "The key insight here is..."
            )
        elif "writing" in cluster_lower or "creative" in cluster_lower:
            return (
                "Here's a thoughtfully crafted response to your request:\n\n"
                "[Detailed, creative content that fully addresses the prompt...]\n\n"
                "I've incorporated the key elements you asked for..."
            )
        else:
            return (
                "I'll address your question comprehensively:\n\n"
                "First, let me provide the direct answer: [detailed response]\n\n"
                "Here's additional context that might be helpful: [supporting information]\n\n"
                "Let me know if you'd like me to elaborate on any aspect."
            )

    def _create_poor_response(self, question: str) -> str:
        """Create a low-quality response that would lose in Arena comparison."""
        poor_patterns = [
            "I'm not sure about that.",
            "I don't have enough information to answer properly.",
            "That's a complex question. You might want to consult an expert.",
            "I cannot provide a detailed answer to that question.",
        ]

        # Select based on question characteristics
        question_lower = question.lower()

        if "how" in question_lower:
            return "There are many ways to do that. It depends on your specific situation."
        elif "why" in question_lower:
            return "There could be several reasons for that."
        elif "what" in question_lower:
            return "I'm not entirely sure what that is. Can you provide more context?"
        elif "code" in question_lower or "program" in question_lower:
            return "# I'm not sure how to implement this\npass"
        else:
            return poor_patterns[0]

