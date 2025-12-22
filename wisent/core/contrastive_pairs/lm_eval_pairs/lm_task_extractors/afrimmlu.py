from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AfrimmluExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "afrimmlu_direct_amh_prompt_1",
    "afrimmlu_direct_eng_prompt_1",
    "afrimmlu_direct_ewe_prompt_1",
    "afrimmlu_direct_fra_prompt_1",
    "afrimmlu_direct_hau_prompt_1",
    "afrimmlu_direct_ibo_prompt_1",
    "afrimmlu_direct_kin_prompt_1",
    "afrimmlu_direct_lin_prompt_1",
    "afrimmlu_direct_lug_prompt_1",
    "afrimmlu_direct_orm_prompt_1",
    "afrimmlu_direct_sna_prompt_1",
    "afrimmlu_direct_sot_prompt_1",
    "afrimmlu_direct_swa_prompt_1",
    "afrimmlu_direct_twi_prompt_1",
    "afrimmlu_direct_wol_prompt_1",
    "afrimmlu_direct_xho_prompt_1",
    "afrimmlu_direct_yor_prompt_1",
    "afrimmlu_direct_zul_prompt_1",
)

class AfrimmluExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Afrimmlu benchmark."""


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

            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            answer = doc.get("answer", "")

            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            else:
                return None

            if not question or not choices or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()
            metadata = {"label": "afrimmlu"}

            return self._build_pair(
                question=question,
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
