from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.config_tools.constants import SENSOR_LAST_OFFSET
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AIME2025Extractor"]

log = setup_logger(__name__)

task_names = ("aime2025",)

class AIME2025Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for AIME 2025 dataset."""


    evaluator_name = "aime"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        # Load AIME 2025 dataset
        docs = self.load_dataset(
            dataset_name="MathArena/aime_2025",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} AIME 2025 examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid AIME 2025 pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            question = str(doc.get("problem", "")).strip()

            if not question:
                log.debug("Skipping: missing problem")
                return None

            # MathArena/aime_2025 has 'answer' as int64 field
            answer_value = doc.get("answer")
            if answer_value is None:
                log.debug("Skipping: missing answer")
                return None

            # Convert answer to int if needed
            if isinstance(answer_value, str):
                correct_int = int(float(answer_value))
            else:
                correct_int = int(answer_value)

            correct = str(correct_int)
            incorrect = str(correct_int + SENSOR_LAST_OFFSET)

            question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {"label": "aime2025"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}")
            return None

