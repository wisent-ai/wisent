from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AgievalLogiQAExtractor"]
_LOG = setup_logger(__name__)


class AgievalLogiQAExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AGIEval LogiQA subtasks.

    Covers agieval_logiqa_en and agieval_logiqa_zh which have the LogiQA schema:
        - context: passage text
        - question: question text
        - options: list of answer strings
        - label: "a", "b", "c", or "d" (index as letter)
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
        Build contrastive pairs from AGIEval LogiQA docs.

        AGIEval LogiQA schema:
            - context: str       — passage text
            - question: str      — question text
            - options: list[str] — answer options
            - label: str         — correct choice as letter ("a", "b", "c", "d")

        Args:
            lm_eval_task_data: lm-eval task instance for an agieval_logiqa subtask.
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
            log.warning("No valid AGIEval LogiQA pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single AGIEval LogiQA doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            options = doc.get("options", [])
            label = str(doc.get("label", "")).strip()

            if not context or not question or not options or not label:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Convert label letter to index
            try:
                label_idx = int(ord(label.lower()) - ord('a'))
            except (ValueError, TypeError):
                log.debug(
                    "Skipping doc: invalid label format",
                    extra={"label": label},
                )
                return None

            if not (0 <= label_idx < len(options)):
                log.debug(
                    "Skipping doc: label index out of range",
                    extra={"label_idx": label_idx, "num_options": len(options)},
                )
                return None

            correct = str(options[label_idx]).strip()
            incorrect = str(options[(label_idx + 1) % len(options)]).strip()

            if not correct or not incorrect:
                log.debug("Skipping doc: empty correct or incorrect answer")
                return None

            # Format the prompt with context and question
            prompt = f"Passage: {context}\nQuestion: {question}"

            metadata = {
                "label": "agieval_logiqa",
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
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
