from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["UnfairExtractor"]
_LOG = setup_logger(__name__)

task_names = ("unfair_tos",)

evaluator_name = "generation"


class UnfairExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Unfair benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Unfair TOS format: source (classification prompt) and target (comma-separated labels or "None")
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip()

            if not source or not target:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            # Use the source as prompt and target as correct response
            # For incorrect response, use a different label
            # Possible unfair TOS labels
            labels = [
                "Limitation of liability",
                "Unilateral termination",
                "Unilateral change",
                "Content removal",
                "Contract by using",
                "Choice of law",
                "Jurisdiction",
                "Arbitration"
            ]

            # Find an incorrect label (different from target)
            # If target is "None", use any label as incorrect
            if target == "None":
                incorrect = labels[0]
            else:
                # Find a label not in target
                target_labels = [l.strip() for l in target.split(",")]
                incorrect = "None"
                for label in labels:
                    if label not in target_labels:
                        incorrect = label
                        break

            metadata = {"label": "unfair"}

            return self._build_pair(
                question=source,
                correct=target,
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
