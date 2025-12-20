from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TruthfulQAMC2Extractor"]
_LOG = setup_logger(__name__)

task_names = ("truthfulqa_mc2",)

class TruthfulQAMC2Extractor(LMEvalBenchmarkExtractor):
    """Extractor for the TruthfulQA_MC2 benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TruthfulQA_MC2 docs.

        TruthfulQA_MC1 schema:
            - question: str
            - mc1_targets: dict
            
        Args:
            lm_eval_task_data: lm-eval task instance for TruthfulQA_MC2.
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
            log.warning("No valid TruthfulQA_MC2 pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA_MC2 doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Uses mc2_targets for richer contrastive signal:
        - Selects the first correct (truthful) answer
        - Selects the first incorrect (false/misleading) answer
        This creates maximum contrast between truth and common misconception.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()

            # Use mc2_targets which has multiple correct/incorrect answers
            mc2_targets = doc.get("mc2_targets")
            if mc2_targets:
                options = mc2_targets["choices"]
                labels = mc2_targets["labels"]
            else:
                # Fallback to mc1_targets
                mc1_targets = doc.get("mc1_targets")
                options = mc1_targets["choices"]
                labels = mc1_targets["labels"]

            if not question or not options or not labels:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Find all correct and incorrect answers
            correct_answers = [opt for opt, lbl in zip(options, labels) if lbl == 1]
            incorrect_answers = [opt for opt, lbl in zip(options, labels) if lbl == 0]

            if not correct_answers or not incorrect_answers:
                log.debug(
                    "Skipping doc - need both correct and incorrect options",
                    extra={"doc": doc},
                )
                return None

            # Use shortest correct (most direct truth) and longest incorrect (most elaborate misconception)
            correct = min(correct_answers, key=len)
            incorrect = max(incorrect_answers, key=len)

            metadata = {
                "label": "truthfulqa_mc2",
            }

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