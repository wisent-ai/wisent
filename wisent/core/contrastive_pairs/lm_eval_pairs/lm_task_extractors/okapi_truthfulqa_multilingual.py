from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["OkapiTruthfulqaMultilingualExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "truthfulqa_ar_mc1", "truthfulqa_ar_mc2", "truthfulqa_bn_mc1", "truthfulqa_bn_mc2",
    "truthfulqa_ca_mc1", "truthfulqa_ca_mc2", "truthfulqa_da_mc1", "truthfulqa_da_mc2",
    "truthfulqa_de_mc1", "truthfulqa_de_mc2", "truthfulqa_es_mc1", "truthfulqa_es_mc2",
    "truthfulqa_eu_mc1", "truthfulqa_eu_mc2", "truthfulqa_fr_mc1", "truthfulqa_fr_mc2",
    "truthfulqa_gu_mc1", "truthfulqa_gu_mc2", "truthfulqa_hi_mc1", "truthfulqa_hi_mc2",
    "truthfulqa_hr_mc1", "truthfulqa_hr_mc2", "truthfulqa_hu_mc1", "truthfulqa_hu_mc2",
    "truthfulqa_hy_mc1", "truthfulqa_hy_mc2", "truthfulqa_id_mc1", "truthfulqa_id_mc2",
    "truthfulqa_it_mc1", "truthfulqa_it_mc2", "truthfulqa_kn_mc1", "truthfulqa_kn_mc2",
    "truthfulqa_ml_mc1", "truthfulqa_ml_mc2", "truthfulqa_mr_mc1", "truthfulqa_mr_mc2",
    "truthfulqa_ne_mc1", "truthfulqa_ne_mc2", "truthfulqa_nl_mc1", "truthfulqa_nl_mc2",
    "truthfulqa_pt_mc1", "truthfulqa_pt_mc2", "truthfulqa_ro_mc1", "truthfulqa_ro_mc2",
    "truthfulqa_ru_mc1", "truthfulqa_ru_mc2", "truthfulqa_sk_mc1", "truthfulqa_sk2",
    "truthfulqa_sr_mc1", "truthfulqa_sr_mc2", "truthfulqa_sv_mc1", "truthfulqa_sv_mc2",
    "truthfulqa_ta_mc1", "truthfulqa_ta_mc2", "truthfulqa_te_mc1", "truthfulqa_te_mc2",
    "truthfulqa_uk_mc1", "truthfulqa_uk_mc2", "truthfulqa_vi_mc1", "truthfulqa_vi_mc2",
    "truthfulqa_zh_mc1", "truthfulqa_zh_mc2"
)

class OkapiTruthfulqaMultilingualExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Okapi/Truthfulqa Multilingual benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Okapi/Truthfulqa Multilingual docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Okapi/Truthfulqa Multilingual.
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
            log.warning("No valid Okapi/Truthfulqa Multilingual pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Okapi/Truthfulqa Multilingual doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: TruthfulQA mc1/mc2 format with labels
            if "question" in doc and ("mc1_choices" in doc or "mc2_choices" in doc):
                question = str(doc.get("question", "")).strip()
                # Prefer mc1 if both exist
                if "mc1_choices" in doc:
                    choices = doc.get("mc1_choices", [])
                    labels = doc.get("mc1_targets_labels", [])
                else:
                    choices = doc.get("mc2_choices", [])
                    labels = doc.get("mc2_targets_labels", [])

                # Find first correct and first incorrect answer
                correct_idx = None
                incorrect_idx = None
                for idx, label in enumerate(labels):
                    if label == 1 and correct_idx is None:
                        correct_idx = idx
                    elif label == 0 and incorrect_idx is None:
                        incorrect_idx = idx
                    if correct_idx is not None and incorrect_idx is not None:
                        break

                if correct_idx is not None and incorrect_idx is not None:
                    correct = choices[correct_idx].strip() if isinstance(choices[correct_idx], str) else str(choices[correct_idx])
                    incorrect = choices[incorrect_idx].strip() if isinstance(choices[incorrect_idx], str) else str(choices[incorrect_idx])

                    if correct and incorrect:
                        metadata = {"label": "okapi/truthfulqa_multilingual"}
                        return self._build_pair(
                            question=question,
                            correct=correct,
                            incorrect=incorrect,
                            metadata=metadata,
                        )
                return None

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

            # Format 3: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "okapi/truthfulqa_multilingual"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "okapi/truthfulqa_multilingual",
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
