from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["OkapiHellaswagMultilingualExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "hellaswag_ar", "hellaswag_bn", "hellaswag_ca", "hellaswag_da", "hellaswag_de",
    "hellaswag_es", "hellaswag_eu", "hellaswag_fr", "hellaswag_gu", "hellaswag_hi",
    "hellaswag_hr", "hellaswag_hu", "hellaswag_hy", "hellaswag_id", "hellaswag_it",
    "hellaswag_kn", "hellaswag_ml", "hellaswag_mr", "hellaswag_ne", "hellaswag_nl",
    "hellaswag_pt", "hellaswag_ro", "hellaswag_ru", "hellaswag_sk", "hellaswag_sr",
    "hellaswag_sv", "hellaswag_ta", "hellaswag_te", "hellaswag_uk", "hellaswag_vi"
)

class OkapiHellaswagMultilingualExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Okapi/Hellaswag Multilingual benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Okapi/Hellaswag Multilingual docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Okapi/Hellaswag Multilingual.
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
            log.warning("No valid Okapi/Hellaswag Multilingual pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Okapi/Hellaswag Multilingual doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: query + choices + gold/label (Hellaswag style)
            if ("query" in doc or "prompt" in doc) and "choices" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                choices = doc.get("choices", [])
                if not isinstance(choices, list):
                    choices = []
                # Try gold, label, or answer field
                answer = doc.get("gold", doc.get("label", doc.get("answer", "")))
                if isinstance(answer, str):
                    if len(answer) == 1 and answer.isalpha():
                        answer_idx = ord(answer.upper()) - ord('A')
                    else:
                        try:
                            answer_idx = int(answer)
                        except (ValueError, TypeError):
                            answer_idx = 0
                else:
                    answer_idx = int(answer) if answer else 0

            # Format 2: question + choices + answer
            elif "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("answer", doc.get("answerKey", ""))
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    answer_idx = int(answer) if answer else 0

            # Format 3: instruction + option_a/b/c/d + answer (MMMLU style)
            elif "instruction" in doc and "option_a" in doc:
                question = str(doc.get("instruction", "")).strip()
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx].strip() if isinstance(choices[answer_idx], str) else str(choices[answer_idx])
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx].strip() if isinstance(choices[incorrect_idx], str) else str(choices[incorrect_idx])

            # Validate that both correct and incorrect are non-empty
            if not correct or not incorrect:
                log.debug(
                    "Skipping doc due to empty correct or incorrect answer",
                    extra={"doc": doc, "correct": correct, "incorrect": incorrect},
                )
                return None

            metadata = {
                "label": "okapi/hellaswag_multilingual",
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
