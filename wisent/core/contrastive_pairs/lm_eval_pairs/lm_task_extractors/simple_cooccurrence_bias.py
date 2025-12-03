from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SimpleCooccurrenceBiasExtractor"]
_LOG = setup_logger(__name__)

task_names = ("simple_cooccurrence_bias",)
class SimpleCooccurrenceBiasExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Simple Cooccurrence Bias benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Simple Cooccurrence Bias docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Simple Cooccurrence Bias.
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
            log.warning("No valid Simple Cooccurrence Bias pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Simple Cooccurrence Bias doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", doc.get("occupation", "unknown")))

        try:
            # Simple Cooccurrence Bias format: sentence + occupation
            # e.g., {"sentence": "The accountant was a", "occupation": "accountant"}
            # The task measures bias by comparing log-likelihoods of gender identifiers

            if "sentence" not in doc:
                log.debug(
                    "Skipping doc due to missing 'sentence' field",
                    extra={"doc": doc},
                )
                return None

            sentence = str(doc.get("sentence", "")).strip()
            if not sentence:
                log.debug(
                    "Skipping doc due to empty sentence",
                    extra={"doc": doc},
                )
                return None

            # Build contrastive pairs with feminine vs masculine identifiers
            # Positive: feminine identifier (woman)
            # Negative: masculine identifier (man)
            prompt = sentence
            positive = prompt + " woman"
            negative = prompt + " man"

            metadata = {
                "label": "simple_cooccurrence_bias",
                "occupation": doc.get("occupation", "unknown"),
            }

            return self._build_pair(
                question=prompt,
                correct=" woman",
                incorrect=" man",
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
