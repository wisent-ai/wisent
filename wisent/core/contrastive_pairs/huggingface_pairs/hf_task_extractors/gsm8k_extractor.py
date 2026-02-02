"""GSM8K (Grade School Math 8K) benchmark extractor."""
from __future__ import annotations
import random
from typing import Any
from wisent.core.cli.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

__all__ = ["GSM8KExtractor"]


class GSM8KExtractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from GSM8K benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "gsm8k"

    def extract_contrastive_pairs(self, limit: int | None = 500) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="gsm8k",
                dataset_config="main",
                split="test",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from GSM8K")
        except Exception as e:
            log.error(f"Failed to load GSM8K: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.
        
        GSM8K schema:
        - question: str (the math word problem)
        - answer: str (solution with reasoning, final answer after ####)
        """
        try:
            question = doc.get("question", "").strip()
            full_answer = doc.get("answer", "").strip()

            if not question or not full_answer:
                return None

            # Extract final numeric answer (after ####)
            if "####" in full_answer:
                reasoning, final_answer = full_answer.split("####", 1)
                final_answer = final_answer.strip()
            else:
                final_answer = full_answer.split()[-1] if full_answer else "0"

            task_prompt = f"""Solve this math problem step by step:

{question}

Provide your solution with reasoning."""

            # Correct answer is the full solution
            correct = full_answer

            # Create incorrect answer with wrong final value
            try:
                num = float(final_answer.replace(",", "").replace("$", ""))
                wrong_num = num + random.choice([1, -1]) * random.randint(1, 10)
                incorrect = f"The answer is {wrong_num}"
            except ValueError:
                incorrect = "I cannot solve this problem."

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="gsm8k",
                metadata={
                    "source": "gsm8k",
                    "final_answer": final_answer,
                },
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None
