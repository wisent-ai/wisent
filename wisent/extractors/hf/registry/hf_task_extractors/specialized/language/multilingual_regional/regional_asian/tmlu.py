from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TmluExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "tmlu_accountant", "tmlu_AST_biology", "tmlu_AST_chemistry", "tmlu_AST_chinese",
    "tmlu_AST_civics", "tmlu_AST_geography", "tmlu_AST_history",
    "tmlu_basic_traditional_chinese_medicine", "tmlu_CAP_biology", "tmlu_CAP_chemistry",
    "tmlu_CAP_chinese", "tmlu_CAP_civics", "tmlu_CAP_earth_science", "tmlu_CAP_geography",
    "tmlu_CAP_history", "tmlu_clinical_psychologist", "tmlu_clinical_traditional_chinese_medicine",
    "tmlu_driving_rule", "tmlu_GSAT_biology", "tmlu_GSAT_chemistry", "tmlu_GSAT_chinese",
    "tmlu_GSAT_civics", "tmlu_GSAT_earth_science", "tmlu_GSAT_geography", "tmlu_GSAT_history",
    "tmlu_lawyer_qualification", "tmlu_nutritionist", "tmlu_taiwan_tourist_resources",
    "tmlu_teacher_qualification", "tmlu_tour_guide", "tmlu_tour_leader",
)

class TmluExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Tmlu benchmark — loads MediaTek-Research/TCEval-v2 (tmmluplus subset)
    since miulab/tmlu was removed from HF."""

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="tmlu")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            ds = load_dataset(
                "MediaTek-Research/TCEval-v2",
                "tmmluplus-accounting",
                split="test",
            )
        except Exception as exc:
            log.error(f"Failed to load TCEval-v2: {exc}")
            return []
        docs = list(ds)[: (max_items * 4 if max_items else None)]
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = "tmlu"
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # TCEval-v2 tmmluplus format: question + A/B/C/D + answer (letter)
            if "question" in doc and "A" in doc and "answer" in doc:
                question = str(doc.get("question", "")).strip()
                choices = [str(doc.get(letter, "")).strip() for letter in ["A", "B", "C", "D"] if doc.get(letter)]
                answer = str(doc.get("answer", "")).strip().upper()
                if question and choices and answer and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer) - ord('A')
                    if 0 <= answer_idx < len(choices):
                        return self._build_pair(
                            question=question,
                            correct=choices[answer_idx],
                            incorrect=choices[(answer_idx + 1) % len(choices)],
                            metadata={"label": "tmlu"},
                        )

            # Try multiple format patterns for question
            question = doc.get("question", doc.get("query", doc.get("input", doc.get("instruction", doc.get("prompt", ""))))).strip()

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

            formatted_question = f"Question: {question}\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "tmlu"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

