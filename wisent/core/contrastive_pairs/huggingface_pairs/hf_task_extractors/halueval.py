from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["HaluEvalExtractor"]

log = setup_logger(__name__)


class HaluEvalExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HaluEval - Hallucination Evaluation Benchmark for LLMs.

    HaluEval is a large-scale hallucination evaluation benchmark containing
    5,000 general user queries with ChatGPT responses and 30,000 task-specific
    examples from QA, knowledge-grounded dialogue, and text summarization.

    For hallucination detection:
    - Positive (correct) = Factually accurate answer (score=1, label=PASS)
    - Negative (incorrect) = Hallucinated answer (score=0, label=FAIL)

    Schema (flowaicom/HaluEval):
        - id: str (unique identifier)
        - passage: str (reference document/context)
        - question: str (query text)
        - answer: str (response to evaluate)
        - label: str (PASS or FAIL)
        - source_ds: str (source dataset identifier)
        - score: int (0=hallucination, 1=accurate)

    The benchmark evaluates whether information provided in answers is factually
    accurate and directly supported by context given in documents.
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "hallucination_detection"

    def __init__(self, source_filter: str | None = None):
        """
        Initialize HaluEval extractor.

        Args:
            source_filter: Optional filter for specific source dataset
        """
        super().__init__()
        self.source_filter = source_filter

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HaluEval examples.

        Creates pairs where:
        - PASS examples (score=1): The answer is factually supported
        - FAIL examples (score=0): The answer contains hallucinations

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="flowaicom/HaluEval",
                split="test",
                limit=max_items * 2 if max_items else None,  # Load extra to find pairs
            )
            log.info(f"Loaded {len(docs)} examples from HaluEval")
        except Exception as e:
            log.error(f"Failed to load HaluEval: {e}")
            return []

        # Separate PASS and FAIL examples
        pass_examples = []
        fail_examples = []

        for doc in docs:
            if self.source_filter:
                source = doc.get("source_ds", "")
                if self.source_filter.lower() not in source.lower():
                    continue

            label = doc.get("label", "")
            score = doc.get("score", -1)

            if label == "PASS" or score == 1:
                pass_examples.append(doc)
            elif label == "FAIL" or score == 0:
                fail_examples.append(doc)

        pairs: list[ContrastivePair] = []

        # Strategy 1: Create pairs from examples with same question/passage
        question_to_examples: dict[str, dict] = {}
        for doc in docs:
            question = doc.get("question", "")
            passage = doc.get("passage", "")
            key = f"{question}|{passage}"

            if key not in question_to_examples:
                question_to_examples[key] = {"pass": [], "fail": []}

            label = doc.get("label", "")
            score = doc.get("score", -1)
            if label == "PASS" or score == 1:
                question_to_examples[key]["pass"].append(doc)
            elif label == "FAIL" or score == 0:
                question_to_examples[key]["fail"].append(doc)

        # Create pairs from matching questions
        for key, examples in question_to_examples.items():
            if examples["pass"] and examples["fail"]:
                pass_doc = examples["pass"][0]
                fail_doc = examples["fail"][0]
                pair = self._create_pair_from_matched(pass_doc, fail_doc)
                if pair:
                    pairs.append(pair)
                    if max_items and len(pairs) >= max_items:
                        break

        # Strategy 2: If not enough pairs, create from individual examples
        if max_items is None or len(pairs) < max_items:
            remaining = (max_items - len(pairs)) if max_items else len(fail_examples)

            for fail_doc in fail_examples[:remaining]:
                pair = self._extract_pair_from_fail_doc(fail_doc)
                if pair:
                    pairs.append(pair)
                    if max_items and len(pairs) >= max_items:
                        break

        if not pairs:
            log.warning("No valid HaluEval pairs extracted")

        return pairs

    def _create_pair_from_matched(
        self,
        pass_doc: dict[str, Any],
        fail_doc: dict[str, Any],
    ) -> ContrastivePair | None:
        """Create a pair from matched PASS and FAIL examples with same context."""
        try:
            passage = pass_doc.get("passage", "")
            question = pass_doc.get("question", "")
            correct_answer = pass_doc.get("answer", "")
            hallucinated_answer = fail_doc.get("answer", "")

            if not all([passage, question, correct_answer, hallucinated_answer]):
                return None

            prompt = self._build_prompt(passage, question)

            metadata = {
                "label": "halueval",
                "source": "flowaicom/HaluEval",
                "source_ds": pass_doc.get("source_ds", ""),
                "pass_id": pass_doc.get("id", ""),
                "fail_id": fail_doc.get("id", ""),
                "is_hallucination_benchmark": True,
                "pair_type": "matched",
            }

            return self._build_pair(
                question=prompt,
                correct=correct_answer,
                incorrect=hallucinated_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error creating matched pair: {exc}", exc_info=True)
            return None

    def _extract_pair_from_fail_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Extract a pair from a FAIL (hallucinated) example.

        For FAIL examples, we use the hallucinated answer as negative
        and generate a plausible correct response.
        """
        try:
            doc_id = doc.get("id", "")
            passage = doc.get("passage", "").strip()
            question = doc.get("question", "").strip()
            hallucinated_answer = doc.get("answer", "").strip()
            source_ds = doc.get("source_ds", "")

            if not all([passage, question, hallucinated_answer]):
                return None

            prompt = self._build_prompt(passage, question)

            # The hallucinated answer is the incorrect response
            incorrect_answer = hallucinated_answer

            # Generate a correct response that acknowledges the context
            correct_answer = self._create_grounded_response(passage, question)

            metadata = {
                "label": "halueval",
                "source": "flowaicom/HaluEval",
                "id": doc_id,
                "source_ds": source_ds,
                "original_label": "FAIL",
                "original_score": 0,
                "is_hallucination_benchmark": True,
                "pair_type": "synthetic_correct",
            }

            return self._build_pair(
                question=prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from fail doc: {exc}", exc_info=True)
            return None

    def _build_prompt(self, passage: str, question: str) -> str:
        """Build a prompt with context and question."""
        return f"""Context: {passage}

Question: {question}

Please answer the question based only on the information provided in the context."""

    def _create_grounded_response(self, passage: str, question: str) -> str:
        """Create a grounded response that stays true to the passage."""
        return (
            f"Based on the provided context, I can tell you that: "
            f"{passage} "
            "I'll only answer based on what's directly stated in the passage."
        )

