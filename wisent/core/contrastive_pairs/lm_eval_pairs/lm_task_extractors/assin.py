from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AssinExtractor"]
_LOG = setup_logger(__name__)

task_names = ("assin_entailment", "assin_paraphrase")

class AssinExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Assin benchmark - entailment task."""


    evaluator_name = "log_likelihoods"
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
            # Assin entailment format has premise, hypothesis, and entailment_judgment
            premise = doc.get("premise", "").strip()
            hypothesis = doc.get("hypothesis", "").strip()
            entailment_judgment = doc.get("entailment_judgment")

            if not premise or not hypothesis or entailment_judgment is None:
                log.debug("Skipping doc due to missing premise/hypothesis/entailment", extra={"doc": doc})
                return None

            # Build the two choices following lm-eval format:
            # Choice 0: "premise, certo? Também, hypothesis"  (no entailment)
            # Choice 1: "premise, certo? Sim, hypothesis"     (entailment)
            choice_0 = f"{premise}, certo? Também, {hypothesis}"
            choice_1 = f"{premise}, certo? Sim, {hypothesis}"

            # entailment_judgment: 0 = no entailment (choice 0), 1 = entailment (choice 1)
            if entailment_judgment == 0:
                correct = choice_0
                incorrect = choice_1
            elif entailment_judgment == 1:
                correct = choice_1
                incorrect = choice_0
            else:
                log.debug("Invalid entailment_judgment value", extra={"entailment_judgment": entailment_judgment})
                return None

            # Use premise as prompt (though the choices contain full text for log-likelihood)
            prompt = premise
            metadata = {"label": "assin_entailment"}

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
