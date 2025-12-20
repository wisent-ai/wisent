from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CevalExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "ceval",
    "ceval-valid",
    "ceval-valid_accountant",
    "ceval-valid_advanced_mathematics",
    "ceval-valid_art_studies",
    "ceval-valid_basic_medicine",
    "ceval-valid_business_administration",
    "ceval-valid_chinese_language_and_literature",
    "ceval-valid_civil_servant",
    "ceval-valid_clinical_medicine",
    "ceval-valid_college_chemistry",
    "ceval-valid_college_economics",
    "ceval-valid_college_physics",
    "ceval-valid_college_programming",
    "ceval-valid_computer_architecture",
    "ceval-valid_computer_network",
    "ceval-valid_discrete_mathematics",
    "ceval-valid_education_science",
    "ceval-valid_electrical_engineer",
    "ceval-valid_environmental_impact_assessment_engineer",
    "ceval-valid_fire_engineer",
    "ceval-valid_high_school_biology",
    "ceval-valid_high_school_chemistry",
    "ceval-valid_high_school_chinese",
    "ceval-valid_high_school_geography",
    "ceval-valid_high_school_history",
    "ceval-valid_high_school_mathematics",
    "ceval-valid_high_school_physics",
    "ceval-valid_high_school_politics",
    "ceval-valid_ideological_and_moral_cultivation",
    "ceval-valid_law",
    "ceval-valid_legal_professional",
    "ceval-valid_logic",
    "ceval-valid_mao_zedong_thought",
    "ceval-valid_marxism",
    "ceval-valid_metrology_engineer",
    "ceval-valid_middle_school_biology",
    "ceval-valid_middle_school_chemistry",
    "ceval-valid_middle_school_geography",
    "ceval-valid_middle_school_history",
    "ceval-valid_middle_school_mathematics",
    "ceval-valid_middle_school_physics",
    "ceval-valid_middle_school_politics",
    "ceval-valid_modern_chinese_history",
    "ceval-valid_operating_system",
    "ceval-valid_physician",
    "ceval-valid_plant_protection",
    "ceval-valid_probability_and_statistics",
    "ceval-valid_professional_tour_guide",
    "ceval-valid_sports_science",
    "ceval-valid_tax_accountant",
    "ceval-valid_teacher_qualification",
    "ceval-valid_urban_and_rural_planner",
    "ceval-valid_veterinary_medicine",
)

class CevalExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Ceval benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Ceval docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Ceval.
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
            log.warning("No valid Ceval pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Ceval doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer
            if "question" in doc and "choices" in doc:
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

            # Format 2: instruction + option_a/b/c/d + answer (MMMLU style)
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

            # Format 3: question + A/B/C/D + answer (C-Eval style)
            elif "question" in doc and "A" in doc:
                question = str(doc.get("question", "")).strip()
                choices = [
                    str(doc.get("A", "")).strip(),
                    str(doc.get("B", "")).strip(),
                    str(doc.get("C", "")).strip(),
                    str(doc.get("D", "")).strip(),
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
                    metadata = {"label": "ceval"}
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
                "label": "ceval",
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
