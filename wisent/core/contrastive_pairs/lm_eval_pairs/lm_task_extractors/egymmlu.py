from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EgyMMLUExtractor"]
_LOG = setup_logger(__name__)


class EgyMMLUExtractor(LMEvalBenchmarkExtractor):
    """Extractor for EgyMMLU benchmark - Egyptian Arabic knowledge assessment.

    This is an MMLU-style multiple choice task testing knowledge across various subjects.
    Format: question + choices (list) + answer (integer index)
    """

    task_names = (
        "egymmlu",
        "egymmlu_accounting",
        "egymmlu_ar_mmlu",
        "egymmlu_ar_mmlu_tasks",
        "egymmlu_arabic_language",
        "egymmlu_arabic_language_(general)",
        "egymmlu_arabic_language_(grammar)",
        "egymmlu_biology",
        "egymmlu_civics",
        "egymmlu_computer_science",
        "egymmlu_driving_test",
        "egymmlu_economics",
        "egymmlu_general_knowledge",
        "egymmlu_geography",
        "egymmlu_global_facts",
        "egymmlu_high_school_european_history",
        "egymmlu_high_school_geography",
        "egymmlu_high_school_government_and_politics",
        "egymmlu_high_school_psychology",
        "egymmlu_high_school_statistics",
        "egymmlu_high_school_world_history",
        "egymmlu_history",
        "egymmlu_human_aging",
        "egymmlu_humanities_tasks",
        "egymmlu_international_law",
        "egymmlu_islamic_studies",
        "egymmlu_jurisprudence",
        "egymmlu_language_tasks",
        "egymmlu_law",
        "egymmlu_logical_fallacies",
        "egymmlu_management",
        "egymmlu_management_ar",
        "egymmlu_marketing",
        "egymmlu_math",
        "egymmlu_mmlu",
        "egymmlu_mmlu_tasks",
        "egymmlu_moral_disputes",
        "egymmlu_moral_scenarios",
        "egymmlu_natural_science",
        "egymmlu_nutrition",
        "egymmlu_other_tasks",
        "egymmlu_philosophy",
        "egymmlu_philosophy_ar",
        "egymmlu_physics",
        "egymmlu_political_science",
        "egymmlu_professional_law",
        "egymmlu_professional_psychology",
        "egymmlu_public_relations",
        "egymmlu_security_studies",
        "egymmlu_social_science",
        "egymmlu_social_sciences_tasks",
        "egymmlu_sociology",
        "egymmlu_stem_tasks",
        "egymmlu_world_religions",
    )
    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Egymmlu docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Egymmlu.
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
            log.warning("No valid Egymmlu pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single EgyMMLU doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("__index_level_0__", "unknown"))

        try:
            # Extract fields - EgyMMLU format
            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            answer = doc.get("answer")

            if not question or not choices or answer is None:
                log.debug(
                    "Skipping doc due to missing fields",
                    extra={"has_question": bool(question), "has_choices": bool(choices), "has_answer": answer is not None},
                )
                return None

            # Validate answer index
            if not isinstance(answer, int) or not (0 <= answer < len(choices)):
                log.debug(
                    "Skipping doc due to invalid answer index",
                    extra={"answer": answer, "num_choices": len(choices)},
                )
                return None

            # Get correct and incorrect answers
            correct = choices[answer]
            # Use the next choice as incorrect (wrapping around)
            incorrect_idx = (answer + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "egymmlu",
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
