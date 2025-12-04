from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["Dbpedia14Extractor"]
_LOG = setup_logger(__name__)

task_names = ("dbpedia_14",)

class Dbpedia14Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for DBpedia 14 - classification task."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="dbpedia_14")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("dbpedia_14", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load dbpedia_14 dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(dataset)})

        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "dbpedia_14"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            content = doc.get("content", "").strip()
            label = doc.get("label")

            if not content or label is None:
                log.debug("Skipping doc due to missing content or label", extra={"doc": doc})
                return None

            # DBpedia 14 categories
            categories = [
                "Company", "EducationalInstitution", "Artist", "Athlete",
                "OfficeHolder", "MeanOfTransportation", "Building",
                "NaturalPlace", "Village", "Animal", "Plant",
                "Album", "Film", "WrittenWork"
            ]

            if not isinstance(label, int) or not (0 <= label < len(categories)):
                log.debug(f"Invalid label: {label}", extra={"doc": doc})
                return None

            correct = categories[label]
            incorrect_idx = (label + 1) % len(categories)
            incorrect = categories[incorrect_idx]

            question = f"Classify the following text into a category:\n\n{content}"
            metadata = {"label": "dbpedia_14"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

