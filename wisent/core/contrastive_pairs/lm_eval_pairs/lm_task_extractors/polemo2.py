from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Polemo2Extractor"]
_LOG = setup_logger(__name__)

task_names = (
    "polemo2",
    "polemo2_in",
    "polemo2_out",
)

class Polemo2Extractor(LMEvalBenchmarkExtractor):
    """Extractor for the Polemo2 benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Polemo2 docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Polemo2.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Polemo2 pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Polemo2 doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Polemo2 format: {sentence: "Polish text...", target: "__label__meta_plus_m"}
        Labels: __label__meta_plus_m (positive), __label__meta_minus_m (negative),
                __label__meta_amb (ambiguous), __label__meta_zero (neutral)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Polemo2 specific format: sentence + target label
            if "sentence" in doc and "target" in doc:
                sentence = str(doc["sentence"]).strip()
                target = str(doc["target"]).strip()

                if not sentence or not target:
                    log.debug("Skipping doc due to empty sentence/target", extra={"doc": doc})
                    return None

                # Map label to sentiment for creating contrastive pairs
                label_mapping = {
                    "__label__meta_plus_m": "positive",
                    "__label__meta_minus_m": "negative",
                    "__label__meta_amb": "ambiguous",
                    "__label__meta_zero": "neutral",
                }

                correct_sentiment = label_mapping.get(target, target.replace("__label__meta_", "").replace("_m", ""))

                # For contrastive pairs, pick an opposite sentiment
                if target == "__label__meta_plus_m":
                    incorrect_sentiment = "negative"
                elif target == "__label__meta_minus_m":
                    incorrect_sentiment = "positive"
                elif target == "__label__meta_amb":
                    incorrect_sentiment = "positive"
                else:  # neutral
                    incorrect_sentiment = "negative"

                question = f"What is the sentiment of this Polish review?\n{sentence[:200]}..." if len(sentence) > 200 else f"What is the sentiment of this Polish review?\n{sentence}"

                metadata = {"label": "polemo2"}

                return self._build_pair(
                    question=question,
                    correct=correct_sentiment,
                    incorrect=incorrect_sentiment,
                    metadata=metadata,
                )

            log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
            return None

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
