from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GaokaoExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "agieval_gaokao_biology",
    "agieval_gaokao_chemistry",
    "agieval_gaokao_chinese",
    "agieval_gaokao_english",
    "agieval_gaokao_geography",
    "agieval_gaokao_history",
    "agieval_gaokao_mathcloze",
    "agieval_gaokao_mathqa",
    "agieval_gaokao_physics",
)

class GaokaoExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AGIEval Gaokao benchmark - Chinese college entrance exam questions."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Gaokao docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Gaokao.
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
            log.warning("No valid Gaokao pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Gaokao doc into a ContrastivePair.

        Gaokao format:
        - query: question text with options
        - choices: list of choice texts
        - gold: list containing the index of correct answer (e.g., [2])

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            query = str(doc.get("query", "")).strip()
            choices = doc.get("choices", [])
            gold = doc.get("gold", [])

            if not query or not choices or not gold:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Gold is a list containing the index
            if isinstance(gold, list) and len(gold) > 0:
                answer_idx = int(gold[0])
            elif isinstance(gold, int):
                answer_idx = gold
            else:
                log.debug("Skipping doc due to invalid gold format", extra={"doc": doc})
                return None

            if not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to invalid answer index", extra={"doc": doc})
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            prompt = f"Question: {query}"

            metadata = {"label": "gaokao"}

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
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
        )
