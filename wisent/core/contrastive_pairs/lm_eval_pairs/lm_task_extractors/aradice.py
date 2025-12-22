"""AraDiCE extractor for Arabic dialect multiple-choice tasks."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AradiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "AraDiCE",
    "AraDiCE_ArabicMMLU_egy",
    "AraDiCE_ArabicMMLU_high_humanities_history_egy",
    "AraDiCE_ArabicMMLU_high_humanities_history_lev",
    "AraDiCE_ArabicMMLU_high_humanities_islamic-studies_egy",
    "AraDiCE_ArabicMMLU_high_humanities_islamic-studies_lev",
    "AraDiCE_ArabicMMLU_high_humanities_philosophy_egy",
    "AraDiCE_ArabicMMLU_high_humanities_philosophy_lev",
    "AraDiCE_ArabicMMLU_high_language_arabic-language_egy",
    "AraDiCE_ArabicMMLU_high_language_arabic-language_lev",
    "AraDiCE_ArabicMMLU_high_social-science_civics_egy",
    "AraDiCE_ArabicMMLU_high_social-science_civics_lev",
    "AraDiCE_ArabicMMLU_high_social-science_economics_egy",
    "AraDiCE_ArabicMMLU_high_social-science_economics_lev",
    "AraDiCE_ArabicMMLU_high_social-science_geography_egy",
    "AraDiCE_ArabicMMLU_high_social-science_geography_lev",
    "AraDiCE_ArabicMMLU_high_stem_biology_egy",
    "AraDiCE_ArabicMMLU_high_stem_biology_lev",
    "AraDiCE_ArabicMMLU_high_stem_computer-science_egy",
    "AraDiCE_ArabicMMLU_high_stem_computer-science_lev",
    "AraDiCE_ArabicMMLU_high_stem_physics_egy",
    "AraDiCE_ArabicMMLU_high_stem_physics_lev",
    "AraDiCE_ArabicMMLU_humanities_egy",
    "AraDiCE_ArabicMMLU_humanities_lev",
    "AraDiCE_ArabicMMLU_language_egy",
    "AraDiCE_ArabicMMLU_language_lev",
    "AraDiCE_ArabicMMLU_lev",
    "AraDiCE_ArabicMMLU_middle_humanities_history_egy",
    "AraDiCE_ArabicMMLU_middle_humanities_history_lev",
    "AraDiCE_ArabicMMLU_middle_humanities_islamic-studies_egy",
    "AraDiCE_ArabicMMLU_middle_humanities_islamic-studies_lev",
    "AraDiCE_ArabicMMLU_middle_language_arabic-language_egy",
    "AraDiCE_ArabicMMLU_middle_language_arabic-language_lev",
    "AraDiCE_ArabicMMLU_middle_other_general-knowledge_egy",
    "AraDiCE_ArabicMMLU_middle_other_general-knowledge_lev",
    "AraDiCE_ArabicMMLU_middle_social-science_civics_egy",
    "AraDiCE_ArabicMMLU_middle_social-science_civics_lev",
    "AraDiCE_ArabicMMLU_middle_social-science_economics_egy",
    "AraDiCE_ArabicMMLU_middle_social-science_economics_lev",
    "AraDiCE_ArabicMMLU_middle_social-science_geography_egy",
    "AraDiCE_ArabicMMLU_middle_social-science_geography_lev",
    "AraDiCE_ArabicMMLU_middle_social-science_social-science_egy",
    "AraDiCE_ArabicMMLU_middle_social-science_social-science_lev",
    "AraDiCE_ArabicMMLU_middle_stem_computer-science_egy",
    "AraDiCE_ArabicMMLU_middle_stem_computer-science_lev",
    "AraDiCE_ArabicMMLU_middle_stem_natural-science_egy",
    "AraDiCE_ArabicMMLU_middle_stem_natural-science_lev",
    "AraDiCE_ArabicMMLU_na_humanities_islamic-studies_egy",
    "AraDiCE_ArabicMMLU_na_humanities_islamic-studies_lev",
    "AraDiCE_ArabicMMLU_na_language_arabic-language-general_egy",
    "AraDiCE_ArabicMMLU_na_language_arabic-language-general_lev",
    "AraDiCE_ArabicMMLU_na_language_arabic-language-grammar_egy",
    "AraDiCE_ArabicMMLU_na_language_arabic-language-grammar_lev",
    "AraDiCE_ArabicMMLU_na_other_driving-test_egy",
    "AraDiCE_ArabicMMLU_na_other_driving-test_lev",
    "AraDiCE_ArabicMMLU_na_other_general-knowledge_egy",
    "AraDiCE_ArabicMMLU_na_other_general-knowledge_lev",
    "AraDiCE_ArabicMMLU_other_egy",
    "AraDiCE_ArabicMMLU_other_lev",
    "AraDiCE_ArabicMMLU_primary_humanities_history_egy",
    "AraDiCE_ArabicMMLU_primary_humanities_history_lev",
    "AraDiCE_ArabicMMLU_primary_humanities_islamic-studies_egy",
    "AraDiCE_ArabicMMLU_primary_humanities_islamic-studies_lev",
    "AraDiCE_ArabicMMLU_primary_language_arabic-language_egy",
    "AraDiCE_ArabicMMLU_primary_language_arabic-language_lev",
    "AraDiCE_ArabicMMLU_primary_other_general-knowledge_egy",
    "AraDiCE_ArabicMMLU_primary_other_general-knowledge_lev",
    "AraDiCE_ArabicMMLU_primary_social-science_geography_egy",
    "AraDiCE_ArabicMMLU_primary_social-science_geography_lev",
    "AraDiCE_ArabicMMLU_primary_social-science_social-science_egy",
    "AraDiCE_ArabicMMLU_primary_social-science_social-science_lev",
    "AraDiCE_ArabicMMLU_primary_stem_computer-science_egy",
    "AraDiCE_ArabicMMLU_primary_stem_computer-science_lev",
    "AraDiCE_ArabicMMLU_primary_stem_math_egy",
    "AraDiCE_ArabicMMLU_primary_stem_math_lev",
    "AraDiCE_ArabicMMLU_primary_stem_natural-science_egy",
    "AraDiCE_ArabicMMLU_primary_stem_natural-science_lev",
    "AraDiCE_ArabicMMLU_prof_humanities_law_egy",
    "AraDiCE_ArabicMMLU_prof_humanities_law_lev",
    "AraDiCE_ArabicMMLU_social-science_egy",
    "AraDiCE_ArabicMMLU_social-science_lev",
    "AraDiCE_ArabicMMLU_stem_egy",
    "AraDiCE_ArabicMMLU_stem_lev",
    "AraDiCE_ArabicMMLU_univ_other_management_egy",
    "AraDiCE_ArabicMMLU_univ_other_management_lev",
    "AraDiCE_ArabicMMLU_univ_social-science_accounting_egy",
    "AraDiCE_ArabicMMLU_univ_social-science_accounting_lev",
    "AraDiCE_ArabicMMLU_univ_social-science_economics_egy",
    "AraDiCE_ArabicMMLU_univ_social-science_economics_lev",
    "AraDiCE_ArabicMMLU_univ_social-science_political-science_egy",
    "AraDiCE_ArabicMMLU_univ_social-science_political-science_lev",
    "AraDiCE_ArabicMMLU_univ_stem_computer-science_egy",
    "AraDiCE_ArabicMMLU_univ_stem_computer-science_lev",
    "AraDiCE_boolq_egy",
    "AraDiCE_boolq_eng",
    "AraDiCE_boolq_lev",
    "AraDiCE_boolq_msa",
    "AraDiCE_egypt_cultural",
    "AraDiCE_jordan_cultural",
    "AraDiCE_lebanon_cultural",
    "AraDiCE_openbookqa_egy",
    "AraDiCE_openbookqa_eng",
    "AraDiCE_openbookqa_lev",
    "AraDiCE_openbookqa_msa",
    "AraDiCE_palestine_cultural",
    "AraDiCE_piqa_egy",
    "AraDiCE_piqa_eng",
    "AraDiCE_piqa_lev",
    "AraDiCE_piqa_msa",
    "AraDiCE_qatar_cultural",
    "AraDiCE_syria_cultural",
    "AraDiCE_truthfulqa_mc1_egy",
    "AraDiCE_truthfulqa_mc1_eng",
    "AraDiCE_truthfulqa_mc1_lev",
    "AraDiCE_truthfulqa_mc1_msa",
    "AraDiCE_winogrande_egy",
    "AraDiCE_winogrande_eng",
    "AraDiCE_winogrande_lev",
    "AraDiCE_winogrande_msa",
)

class AradiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AraDiCE benchmark - Arabic dialect multiple-choice tasks."""


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
            # Try multiple format patterns for question
            question = doc.get("question", doc.get("query", doc.get("input", doc.get("instruction", doc.get("prompt", doc.get("sentence", "")))))).strip()

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

            # Handle option1/option2 format (winogrande-style)
            if not choices and "option1" in doc:
                choices = [
                    str(doc.get("option1", "")).strip(),
                    str(doc.get("option2", "")).strip(),
                ]
                choices = [c for c in choices if c]

            # Handle binary tasks with target but no explicit choices (e.g., boolq)
            # For Arabic boolq: target is "نعم" or "لا"
            if not choices and "target" in doc:
                target = str(doc.get("target", "")).strip()
                if target in ["نعم", "لا"]:  # Arabic Yes/No
                    choices = ["لا", "نعم"]  # No, Yes
                elif target in ["yes", "no", "true", "false"]:
                    choices = ["no", "yes"]  # English

            # Try multiple format patterns for answer
            answer = doc.get("answer", doc.get("label", doc.get("target", doc.get("gold", None))))

            # Handle different answer formats
            if isinstance(answer, str):
                if len(answer) == 1 and answer.isalpha():
                    # Answer is like 'A', 'B', 'C', 'D'
                    answer_idx = ord(answer.upper()) - ord('A')
                elif answer.isdigit():
                    # Answer is like '0', '1', '2', '3'
                    answer_idx = int(answer)
                else:
                    # Try to match answer text against choices
                    if choices:
                        try:
                            answer_idx = choices.index(answer)
                        except ValueError:
                            # For binary tasks, default to 1 (yes) if answer looks positive
                            if len(choices) == 2:
                                answer_idx = 1
                            else:
                                return None
                    else:
                        return None
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
            metadata = {"label": "aradice"}

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
