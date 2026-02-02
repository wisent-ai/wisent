"""bABI and SciQ benchmark extractors."""
from __future__ import annotations
from typing import Any
from wisent.core.cli.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

__all__ = ["BABIExtractor", "SciQExtractor"]


class BABIExtractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from bABI benchmark."""

    def __init__(self, task: str = "qa1"):
        super().__init__()
        self.name = "babi"
        self.task = task

    def extract_contrastive_pairs(self, limit: int | None = 500) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        # bABI has multiple tasks (qa1-qa20)
        task_configs = [f"en-10k-{self.task}"] if self.task else ["en-10k-qa1"]
        
        for config in task_configs:
            try:
                docs = self.load_dataset(
                    dataset_name="facebook/babi_qa",
                    dataset_config=config,
                    split="test",
                    limit=limit,
                )
                log.info(f"Loaded {len(docs)} examples from bABI ({config})")
                
                for doc in docs:
                    pair = self._extract_pair_from_doc(doc)
                    if pair:
                        pairs.append(pair)
                        if limit and len(pairs) >= limit:
                            return pairs
            except Exception as e:
                log.warning(f"Failed to load bABI config {config}: {e}")
                continue

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            story = doc.get("story", {})
            
            # bABI format: story with sentences and questions
            texts = story.get("text", [])
            types = story.get("type", [])
            answers = story.get("answer", [])
            
            if not texts:
                return None
            
            # Build context from non-question sentences
            context_parts = []
            question = ""
            correct = ""
            
            for i, (text, typ, ans) in enumerate(zip(texts, types, answers)):
                if typ == 1:  # Question
                    question = text
                    correct = ans
                else:  # Context
                    context_parts.append(text)
            
            if not question or not correct:
                return None

            context = " ".join(context_parts)
            
            task_prompt = f"""Based on the following story, answer the question.

Story: {context}

Question: {question}

Answer:"""

            incorrect = "I don't know."

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="babi",
                metadata={"source": "facebook/babi_qa", "task": self.task},
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None


class SciQExtractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from SciQ benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "sciq"

    def extract_contrastive_pairs(self, limit: int | None = 500) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="allenai/sciq",
                split="test",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from SciQ")
        except Exception as e:
            log.error(f"Failed to load SciQ: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            question = doc.get("question", "").strip()
            support = doc.get("support", "").strip()
            correct_answer = doc.get("correct_answer", "").strip()
            distractor1 = doc.get("distractor1", "").strip()
            distractor2 = doc.get("distractor2", "").strip()
            distractor3 = doc.get("distractor3", "").strip()
            
            if not question or not correct_answer:
                return None

            # Build multiple choice format
            choices = [correct_answer, distractor1, distractor2, distractor3]
            
            if support:
                task_prompt = f"""Based on the following support text, answer the science question.

Support: {support[:1000]}

Question: {question}

Options:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""
            else:
                task_prompt = f"""Answer the following science question.

Question: {question}

Options:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""

            correct = f"A. {correct_answer}"
            incorrect = f"B. {distractor1}"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="sciq",
                metadata={"source": "allenai/sciq"},
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None
