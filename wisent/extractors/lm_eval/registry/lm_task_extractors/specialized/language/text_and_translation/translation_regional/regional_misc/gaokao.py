from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

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
        *,
        train_ratio: float,
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
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)

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
            gold = doc.get("gold")

            if not query or not choices or gold is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Gold may be an int (e.g. 2) or a list containing the index (e.g. [2])
            if isinstance(gold, list):
                if len(gold) == 0:
                    log.debug("Skipping doc due to empty gold list", extra={"doc": doc})
                    return None
                answer_idx = int(gold[0])
            elif isinstance(gold, (int, float)):
                answer_idx = int(gold)
            else:
                log.debug("Skipping doc due to invalid gold format", extra={"doc": doc})
                return None

            if not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to invalid answer index", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            if not correct or not incorrect:
                log.debug("Skipping doc: empty correct or incorrect answer")
                return None

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
