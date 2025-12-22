from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["XNLIExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "xnli_ar",
    "xnli_bg",
    "xnli_de",
    "xnli_el",
    "xnli_en",
    "xnli_es",
    "xnli_fr",
    "xnli_hi",
    "xnli_ru",
    "xnli_sw",
    "xnli_th",
    "xnli_tr",
    "xnli_ur",
    "xnli_vi",
    "xnli_zh",
)

class XNLIExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the XNLI benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from XNLI docs.

        XNLI schema:
            - premise: str
            - hypothesis: str
            - label: 0 or 1 or 2
            
        Args:
            lm_eval_task_data: lm-eval task instance for XNLI.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

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
            log.warning("No valid XNLI pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single XNLI doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            premise = str(doc.get("premise", "")).strip()
            hypothesis = str(doc.get("hypothesis", "")).strip()
            label = doc.get("label")

            if not premise or not hypothesis or label not in {0, 1, 2}:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None
            
            labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
            correct = labels[label]
            incorrect = labels[(label+1)%3]
            
            prompt = f"Decide the relationship of the hypothesis '{hypothesis}' to the premise '{premise}"

            metadata = {
                "label": "xnli",
            }

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