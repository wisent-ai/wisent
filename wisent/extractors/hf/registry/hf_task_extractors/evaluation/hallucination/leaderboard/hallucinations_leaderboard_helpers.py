"""HallucinationsLeaderboard extractor helpers (HaluEval, QA, hallucination response)."""
from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)


class HallucinationsLeaderboardHelperMixin:
    """Mixin providing HaluEval/QA loading methods."""

    def _load_halueval_pairs(self, limit: int | None) -> list[ContrastivePair]:
        """Load pairs from HaluEval dataset."""
        try:
            docs = self.load_dataset(
                dataset_name="pminervini/HaluEval",
                dataset_config="qa_samples",
                split="data",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from HaluEval")
        except Exception as e:
            log.error(f"Failed to load HaluEval from HuggingFace: {e}")
            log.error("HallucinationsLeaderboard requires pminervini/HaluEval dataset. No synthetic data available.")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_halueval_pair(doc)
            if pair is not None:
                pairs.append(pair)
                if limit is not None and len(pairs) >= limit:
                    break

        return pairs

    def _extract_halueval_pair(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Extract a contrastive pair from HaluEval."""
        try:
            knowledge = doc.get("knowledge", "").strip()
            question = doc.get("question", "").strip()
            hallucinated = doc.get("hallucinated_answer", "").strip()
            correct = doc.get("right_answer", "").strip()

            if not question:
                return None

            # Build the prompt with knowledge context
            if knowledge:
                prompt = f"Context: {knowledge}\n\nQuestion: {question}\n\nProvide an answer based only on the given context."
            else:
                prompt = f"Question: {question}\n\nProvide a factual answer."

            # Positive = correct answer
            if correct:
                correct_response = correct
            else:
                return None

            # Negative = hallucinated answer
            if hallucinated:
                incorrect_response = hallucinated
            else:
                incorrect_response = self._create_hallucinated_response(question)

            metadata = {
                "label": "hallucinations_leaderboard",
                "source": "pminervini/HaluEval",
                "task": "halueval",
                "is_hallucination_benchmark": True,
                "has_knowledge_context": bool(knowledge),
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting HaluEval pair: {exc}")
            return None

    def _load_qa_pairs(self, task: str, limit: int | None) -> list[ContrastivePair]:
        """Load pairs from NQ Open or TriviaQA."""
        try:
            if task == "nq_open":
                docs = self.load_dataset(
                    dataset_name="nq_open",
                    split="validation",
                    limit=limit,
                )
            else:  # triviaqa
                docs = self.load_dataset(
                    dataset_name="trivia_qa",
                    dataset_config="rc.nocontext",
                    split="validation",
                    limit=limit,
                )
            log.info(f"Loaded {len(docs)} examples from {task}")
        except Exception as e:
            log.error(f"Failed to load {task}: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_qa_pair(doc, task)
            if pair is not None:
                pairs.append(pair)
                if limit is not None and len(pairs) >= limit:
                    break

        return pairs

    def _extract_qa_pair(self, doc: dict[str, Any], task: str) -> ContrastivePair | None:
        """Extract a contrastive pair from open-domain QA."""
        try:
            question = doc.get("question", "").strip()
            answers = doc.get("answer", [])

            if not question or not answers:
                return None

            # Handle different answer formats
            if isinstance(answers, str):
                correct_answer = answers
            elif isinstance(answers, list) and answers:
                correct_answer = answers[0]
            elif isinstance(answers, dict):
                correct_answer = answers.get("value", answers.get("text", ""))
            else:
                return None

            prompt = f"Question: {question}\n\nProvide a brief, factual answer."

            # Positive = correct answer
            correct_response = correct_answer

            # Negative = hallucinated answer
            incorrect_response = self._create_hallucinated_response(question)

            metadata = {
                "label": "hallucinations_leaderboard",
                "source": f"{task}",
                "task": task,
                "is_hallucination_benchmark": True,
                "is_open_domain_qa": True,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting QA pair: {exc}")
            return None

    def _create_hallucinated_response(self, question: str) -> str:
        """Create a hallucinated response for a question."""
        return (
            "Based on my knowledge, I believe the answer involves several factors "
            "that aren't commonly discussed. According to recent studies from 2024, "
            "experts have revised their understanding significantly. The true answer "
            "is more nuanced than typically reported and requires considering "
            "multiple perspectives that contradict conventional wisdom."
        )

