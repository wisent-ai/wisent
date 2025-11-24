from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["TwentyNewsgroupsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("20_newsgroups",)

evaluator_name = "generation"


class TwentyNewsgroupsExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Twenty Newsgroups benchmark - text classification task."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="20_newsgroups")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        from datasets import load_dataset
        try:
            dataset = load_dataset("SetFit/20_newsgroups", split="train")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load 20_newsgroups dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "20_newsgroups"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Extract text and label
            text = doc.get("text", "").strip()
            label = doc.get("label", None)
            label_text = doc.get("label_text", "").strip()

            if not text or (label is None and not label_text):
                log.debug("Skipping doc due to missing text or label", extra={"doc": doc})
                return None

            # Get correct answer
            correct = label_text if label_text else str(label)

            # For incorrect, use a different newsgroup category
            # Common 20 newsgroups categories
            categories = [
                "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
                "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
                "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
                "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
                "sci.space", "soc.religion.christian", "talk.politics.guns",
                "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"
            ]

            # Find an incorrect category
            incorrect = next((cat for cat in categories if cat != correct), "misc.forsale")

            prompt = f"Classify the following text into a newsgroup category:\n\n{text}"
            metadata = {"label": "20_newsgroups"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
