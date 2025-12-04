from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["BabilongExtractor"]
_LOG = setup_logger(__name__)

task_names = ("babilong",)

class BabilongExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for the Babilong benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Babilong docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task="babilong")

        max_items = self._normalize_limit(limit)

        # Load dataset using base class method
        docs = self.load_dataset(
            dataset_name="RMT-team/babilong",
            dataset_config="qa1",
            split="0k",
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
            log.warning("No valid Babilong pairs extracted", extra={"task": "babilong"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Babilong doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        Schema: {'context': str, 'input': str, 'positive_outputs': list, 'negative_outputs': list}
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:

            input = doc.get("input", "").strip()
            question = doc.get("question", "").strip()
            correct = doc.get("target", "").strip()

            if not input or not question or not correct:
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None
            
            incorrect = "garden" if correct == "bathroom" else "bathroom"

            # Format prompt with context and question
            prompt = f"Context: {input}\n\nQuestion: {question}\nA. {incorrect}\nB. {correct}"

            metadata = {"label": "babilong"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

