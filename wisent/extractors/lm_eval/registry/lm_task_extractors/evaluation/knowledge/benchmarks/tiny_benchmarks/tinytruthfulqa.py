from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TinytruthfulqaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "tinyTruthfulQA",
    "tinyTruthfulQA_mc1",
)

class TinytruthfulqaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Tinytruthfulqa benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple format patterns for question
            question = doc.get("question", doc.get("query", doc.get("input", doc.get("instruction", doc.get("prompt", ""))))).strip()

            # tinyTruthfulQA mc1 format: mc1_targets has choices + labels
            if "mc1_targets" in doc:
                mc1 = doc.get("mc1_targets", {})
                if isinstance(mc1, dict):
                    choices_list = mc1.get("choices", [])
                    labels = mc1.get("labels", [])
                    if choices_list and labels and len(choices_list) == len(labels):
                        try:
                            correct_idx = labels.index(1)
                            correct = str(choices_list[correct_idx]).strip()
                            incorrect_idx = (correct_idx + 1) % len(choices_list)
                            incorrect = str(choices_list[incorrect_idx]).strip()
                            return self._build_pair(
                                question=question,
                                correct=correct,
                                incorrect=incorrect,
                                metadata={"label": "tinytruthfulqa"},
                            )
                        except (ValueError, IndexError):
                            return None
                return None

            # Try multiple format patterns for choices
            choices = doc.get("choices", doc.get("options", doc.get("answers", [])))

            # Handle option_a/b/c/d format
            if not choices and "option_a" in doc:
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]

            # Try multiple format patterns for answer
            answer = doc.get("answer", doc.get("label", doc.get("target", None)))

            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                return None

            if not question or not choices or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()
            metadata = {"label": "tinytruthfulqa"}

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
