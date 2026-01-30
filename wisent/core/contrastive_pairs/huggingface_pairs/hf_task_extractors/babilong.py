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

    # Load from multiple qa configs to get more samples (each config has 100 samples)
    QA_CONFIGS = ["qa1", "qa2", "qa3", "qa4", "qa5"]

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
        pairs: list[ContrastivePair] = []

        # Load from multiple qa configs to get more samples
        for qa_config in self.QA_CONFIGS:
            if max_items is not None and len(pairs) >= max_items:
                break

            # Calculate how many more we need
            remaining = (max_items - len(pairs)) if max_items else None

            try:
                docs = self.load_dataset(
                    dataset_name="RMT-team/babilong",
                    dataset_config=qa_config,
                    split="0k",
                    limit=remaining,
                )

                log.info("Extracting contrastive pairs", extra={"doc_count": len(docs), "config": qa_config})

                for doc in docs:
                    pair = self._extract_pair_from_doc(doc, qa_config)
                    if pair is not None:
                        pairs.append(pair)
                        if max_items is not None and len(pairs) >= max_items:
                            break
            except Exception as e:
                log.warning(f"Failed to load config {qa_config}: {e}")

        if not pairs:
            log.warning("No valid Babilong pairs extracted", extra={"task": "babilong"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], qa_config: str = "qa1") -> ContrastivePair | None:
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

            # Generate an incorrect answer based on the correct one
            # Babilong answers are typically locations
            possible_locations = ["bathroom", "garden", "kitchen", "bedroom", "hallway", "office"]
            incorrect = "garden" if correct not in ["garden"] else "bathroom"
            for loc in possible_locations:
                if loc != correct:
                    incorrect = loc
                    break

            # Format prompt with context and question
            prompt = f"Context: {input}\n\nQuestion: {question}\nA. {incorrect}\nB. {correct}"

            metadata = {"label": f"babilong_{qa_config}"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

