from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["BanglaMmluExtractor"]
_LOG = setup_logger(__name__)

task_names = ("bangla_mmlu",)

class BanglaMmluExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for the Bangla Mmlu benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Bangla Mmlu docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task="bangla_mmlu")

        max_items = self._normalize_limit(limit)

        # Load dataset using base class method
        docs = self.load_dataset(
            dataset_name="hishab/titulm-bangla-mmlu",
            dataset_config="all",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Bangla Mmlu pairs extracted", extra={"task": "bangla_mmlu"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Bangla Mmlu doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", "").strip()
            options = doc.get("options", [])
            answer = doc.get("answer", "").strip()

            if not question or not options or not answer:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
            answer_idx = letter_to_idx.get(answer)

            if answer_idx is None or answer_idx >= len(options):
                log.debug("Invalid answer index", extra={"doc": doc})
                return None

            correct = options[answer_idx]
            # Shift by one: A->B, B->C, C->D, D->A
            incorrect_idx = (answer_idx + 1) % len(options)
            incorrect = options[incorrect_idx]

            formatted_question = f"Question: {question}"

            metadata = {
                "label": "bangla_mmlu",
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

