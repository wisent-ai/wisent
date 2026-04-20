"""SQuAD v2, DROP, and PubMedQA benchmark extractors."""
from __future__ import annotations
from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

__all__ = ["SQuADv2Extractor", "DROPExtractor", "PubMedQAHFExtractor"]


class SQuADv2Extractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from SQuAD v2 benchmark."""
    evaluator_name = "generation"

    def __init__(self, context_max_length: int | None = None):
        super().__init__()
        self.name = "squadv2"
        if context_max_length is None:
            from wisent.core.utils.config_tools.constants import EXTRACTOR_CONTEXT_MAX_LENGTH
            context_max_length = EXTRACTOR_CONTEXT_MAX_LENGTH
        self._context_max_length = context_max_length

    def extract_contrastive_pairs(self, limit: int | None = None) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_all_splits(dataset_name="rajpurkar/squad_v2")
            if limit:
                docs = docs[:limit]
            log.info(f"Loaded {len(docs)} examples from SQuAD v2")
        except Exception as e:
            log.error(f"Failed to load SQuAD v2: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            context = doc.get("context", "").strip()
            question = doc.get("question", "").strip()
            answers = doc.get("answers", {})
            
            answer_texts = answers.get("text", [])
            if not answer_texts:
                # Unanswerable question
                correct = "The question cannot be answered based on the given context."
                incorrect = "Based on the context, the answer is [incorrect guess]."
            else:
                correct = answer_texts[0]
                incorrect = "I don't know the answer."

            if not context or not question:
                return None

            task_prompt = f"""Read the following context and answer the question.

Context: {context[:self._context_max_length]}

Question: {question}

Answer:"""

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="squadv2",
                metadata={"source": "rajpurkar/squad_v2"},
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None


class DROPExtractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from DROP benchmark."""
    evaluator_name = "generation"

    def __init__(self, context_max_length: int | None = None):
        super().__init__()
        self.name = "drop"
        if context_max_length is None:
            from wisent.core.utils.config_tools.constants import EXTRACTOR_CONTEXT_MAX_LENGTH
            context_max_length = EXTRACTOR_CONTEXT_MAX_LENGTH
        self._context_max_length = context_max_length

    def extract_contrastive_pairs(self, limit: int | None = None) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="ucinlp/drop",
                split="validation",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from DROP")
        except Exception as e:
            log.error(f"Failed to load DROP: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            passage = doc.get("passage", "").strip()
            question = doc.get("question", "").strip()
            answers = doc.get("answers_spans", {})
            
            answer_spans = answers.get("spans", [])
            if answer_spans:
                correct = answer_spans[0]
            else:
                return None

            if not passage or not question:
                return None

            task_prompt = f"""Read the passage and answer the question (may require reasoning or calculation).

Passage: {passage[:self._context_max_length]}

Question: {question}

Answer:"""

            incorrect = "I cannot determine the answer."

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="drop",
                metadata={"source": "ucinlp/drop"},
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None


class PubMedQAHFExtractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from PubMedQA via qiaojin/PubMedQA (no scripts)."""
    evaluator_name = "log_likelihoods"

    def __init__(self):
        super().__init__()
        self.name = "pubmedqa"

    def extract_contrastive_pairs(self, limit: int | None = None) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []
        try:
            docs = self.load_dataset(
                dataset_name="qiaojin/PubMedQA",
                dataset_config="pqa_labeled",
                split="train",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from PubMedQA")
        except Exception as e:
            log.error(f"Failed to load PubMedQA: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)
        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            context_obj = doc.get("context", {})
            contexts = context_obj.get("contexts", []) if isinstance(context_obj, dict) else []
            question = str(doc.get("question", "")).strip()
            final_decision = str(doc.get("final_decision", "")).strip()
            if not contexts or not question or not final_decision:
                return None
            formatted = " ".join(s.strip() for s in contexts if isinstance(s, str) and s.strip())
            prompt = f"Abstract: {formatted}\nQuestion: {question}"
            correct = final_decision
            incorrect = "yes" if correct == "no" else "no"
            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)
            return ContrastivePair(
                prompt=prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="pubmedqa",
                metadata={"source": "qiaojin/PubMedQA"},
            )
        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None
