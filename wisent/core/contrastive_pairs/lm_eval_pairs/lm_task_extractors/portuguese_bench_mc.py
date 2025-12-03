from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PortugueseBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "assin_entailment",
    "assin_paraphrase",
)
class PortugueseBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Portuguese Bench multiple-choice benchmarks (ASSIN)."""


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
            log.warning("No valid Portuguese Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            premise = doc.get("premise", "").strip()
            hypothesis = doc.get("hypothesis", "").strip()

            if not premise or not hypothesis:
                log.debug("Skipping doc due to missing premise/hypothesis", extra={"doc": doc})
                return None

            # For entailment task
            if "entailment_judgment" in doc:
                judgment = doc.get("entailment_judgment")
                # judgment: 0 = not entailment, 1 = entailment
                if judgment == 1:
                    correct = f"{premise}, certo? Sim, {hypothesis}"
                    incorrect = f"{premise}, certo? Também, {hypothesis}"
                else:
                    correct = f"{premise}, certo? Também, {hypothesis}"
                    incorrect = f"{premise}, certo? Sim, {hypothesis}"

            # For paraphrase task
            elif "similarity" in doc:
                similarity = doc.get("similarity", 0)
                # If similar (paraphrase), use "Sim", else use "Também"
                if similarity >= 3:  # Threshold for paraphrase
                    correct = f"{premise}, certo? Sim, {hypothesis}"
                    incorrect = f"{premise}, certo? Também, {hypothesis}"
                else:
                    correct = f"{premise}, certo? Também, {hypothesis}"
                    incorrect = f"{premise}, certo? Sim, {hypothesis}"
            else:
                log.debug("Skipping doc due to missing judgment/similarity", extra={"doc": doc})
                return None

            formatted_question = f"Premise: {premise}\nHypothesis: {hypothesis}"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="portuguese_bench_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
