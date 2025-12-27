from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["HallucinationsLeaderboardExtractor"]

log = setup_logger(__name__)

# Tasks included in the Hallucinations Leaderboard
HALLUCINATION_TASKS = [
    "nq_open",  # Natural Questions Open
    "triviaqa",  # TriviaQA
    "truthfulqa_mc1",  # TruthfulQA MC1
    "truthfulqa_mc2",  # TruthfulQA MC2
    "truthfulqa_gen",  # TruthfulQA Generation
    "selfcheckgpt",  # Self-consistency check
    "halueval_qa",  # HaluEval QA
    "halueval_dialog",  # HaluEval Dialog
    "halueval_summarization",  # HaluEval Summarization
]


class HallucinationsLeaderboardExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for the Hallucinations Leaderboard - comprehensive hallucination evaluation.

    The Hallucinations Leaderboard (Edinburgh University) evaluates LLMs against
    benchmarks specifically designed to assess hallucination-related issues.
    It leverages the EleutherAI Language Model Evaluation Harness.

    Tasks include:
    - Closed-book Open-domain QA: NQ Open, TriviaQA
    - TruthfulQA: MC1, MC2, and Generative
    - Hallucination detection: SelfCheckGPT, HaluEval

    This extractor creates contrastive pairs from multiple hallucination-related
    datasets to evaluate model faithfulness and factual accuracy.

    For hallucination evaluation:
    - Positive (correct) = Factually accurate, consistent response
    - Negative (incorrect) = Hallucinated, inconsistent response
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "hallucination_detection"

    def __init__(self, task: str | None = None):
        """
        Initialize Hallucinations Leaderboard extractor.

        Args:
            task: Optional specific task (nq_open, triviaqa, truthfulqa_*, halueval_*)
        """
        super().__init__()
        self.task = task

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Hallucinations Leaderboard tasks.

        For hallucination evaluation:
        - Positive (correct) = Factually accurate response
        - Negative (incorrect) = Hallucinated response

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        pairs: list[ContrastivePair] = []

        # If specific task requested, only load that task
        tasks_to_load = [self.task] if self.task else ["truthfulqa", "halueval"]

        for task in tasks_to_load:
            task_pairs = self._load_task_pairs(task, max_items)
            pairs.extend(task_pairs)

            if max_items is not None and len(pairs) >= max_items:
                pairs = pairs[:max_items]
                break

        if not pairs:
            log.warning("No valid Hallucinations Leaderboard pairs extracted")

        return pairs

    def _load_task_pairs(
        self, task: str, limit: int | None
    ) -> list[ContrastivePair]:
        """Load pairs from a specific hallucination task."""

        if task in ["truthfulqa", "truthfulqa_gen", "truthfulqa_mc1", "truthfulqa_mc2"]:
            return self._load_truthfulqa_pairs(limit)
        elif task in ["halueval", "halueval_qa", "halueval_dialog", "halueval_summarization"]:
            return self._load_halueval_pairs(limit)
        elif task in ["nq_open", "triviaqa"]:
            return self._load_qa_pairs(task, limit)
        else:
            log.warning(f"Unknown task: {task}")
            return []

    def _load_truthfulqa_pairs(self, limit: int | None) -> list[ContrastivePair]:
        """Load pairs from TruthfulQA dataset."""
        try:
            docs = self.load_dataset(
                dataset_name="truthfulqa/truthful_qa",
                dataset_config="generation",
                split="validation",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from TruthfulQA")
        except Exception as e:
            log.error(f"Failed to load TruthfulQA: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_truthfulqa_pair(doc)
            if pair is not None:
                pairs.append(pair)
                if limit is not None and len(pairs) >= limit:
                    break

        return pairs

    def _extract_truthfulqa_pair(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Extract a contrastive pair from TruthfulQA."""
        try:
            question = doc.get("question", "").strip()
            best_answer = doc.get("best_answer", "").strip()
            correct_answers = doc.get("correct_answers", [])
            incorrect_answers = doc.get("incorrect_answers", [])
            category = doc.get("category", "")

            if not question:
                return None

            # Build the prompt
            prompt = f"Question: {question}\n\nProvide a truthful and accurate answer."

            # Positive = correct answer
            if best_answer:
                correct_response = best_answer
            elif correct_answers:
                correct_response = correct_answers[0]
            else:
                return None

            # Negative = incorrect (hallucinated) answer
            if incorrect_answers:
                incorrect_response = incorrect_answers[0]
            else:
                incorrect_response = self._create_hallucinated_response(question)

            metadata = {
                "label": "hallucinations_leaderboard",
                "source": "truthfulqa/truthful_qa",
                "task": "truthfulqa",
                "category": category,
                "is_hallucination_benchmark": True,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting TruthfulQA pair: {exc}")
            return None

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

