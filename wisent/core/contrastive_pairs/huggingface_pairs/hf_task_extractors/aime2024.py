from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AIME2024Extractor"]

log = setup_logger(__name__)

task_names = ("aime2024",)

class AIME2024Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for AIME 2024 dataset."""


    evaluator_name = "aime"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        # Load AIME 2024 dataset
        docs = self.load_dataset(
            dataset_name="Maxwell-Jia/AIME_2024",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} AIME 2024 examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid AIME 2024 pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            # Fields are capitalized in Maxwell-Jia/AIME_2024: Problem, Answer, Solution
            problem = doc.get("problem", "").strip()
            correct = doc.get("answer", "").strip()

            if not problem or not answer:
                log.debug("Skipping: missing problem or answer")
                return None

            incorrect = str(int(correct) + 1)

            question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {"label": "aime2024"}

            return self._build_pair(
                question=question,
                correct=correct_answer,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}")
            return None

