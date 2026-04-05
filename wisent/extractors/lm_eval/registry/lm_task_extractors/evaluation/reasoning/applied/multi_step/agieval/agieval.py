from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AgievalExtractor"]
_LOG = setup_logger(__name__)


class AgievalExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AGIEval benchmark subtasks.

    Covers all agieval_* subtasks that share the common aqua-rat schema:
        - query:   the question text
        - choices: list of answer strings
        - gold:    integer index of the correct answer

    Examples: agieval_gaokao_biology, agieval_gaokao_chemistry,
              agieval_logiqa_en, agieval_lsat_ar, agieval_sat_en, etc.
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: "ConfigurableTask",
        limit: int | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from AGIEval docs.

        AGIEval schema (shared aqua-rat base):
            - query:   str       — the full question text
            - choices: list[str] — answer options
            - gold:    int       — index of the correct choice

        Args:
            lm_eval_task_data: lm-eval task instance for an agieval subtask.
            limit: Optional maximum number of pairs to produce.
            train_ratio: Fraction of docs used for training split.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, train_ratio=train_ratio)

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
            log.warning("No valid AGIEval pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single AGIEval doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            query = str(doc.get("query", "")).strip()
            choices = doc.get("choices", [])
            gold = doc.get("gold")

            if not query or not choices or gold is None:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Handle both formats: integer (standard agieval) and list (gaokao variant)
            if isinstance(gold, list):
                if len(gold) == 0:
                    log.debug(
                        "Skipping doc due to empty gold list",
                        extra={"doc": doc},
                    )
                    return None
                gold_idx = int(gold[0])
            else:
                gold_idx = int(gold)

            if not (0 <= gold_idx < len(choices)):
                log.debug(
                    "Skipping doc: gold index out of range",
                    extra={"gold": gold_idx, "num_choices": len(choices)},
                )
                return None

            correct = str(choices[gold_idx]).strip()
            incorrect = str(choices[(gold_idx + 1) % len(choices)]).strip()

            if not correct or not incorrect:
                log.debug("Skipping doc: empty correct or incorrect answer")
                return None

            metadata = {
                "label": "agieval",
            }

            return self._build_pair(
                question=query,
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
            metadata=metadata,
        )
