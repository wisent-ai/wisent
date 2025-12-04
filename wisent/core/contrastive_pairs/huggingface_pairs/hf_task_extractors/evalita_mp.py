from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EvalitaMpExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "evalita-mp",
    "evalita-mp_at", "evalita-mp_at_prompt-1", "evalita-mp_at_prompt-2", "evalita-mp_at_prompt-3",
    "evalita-mp_at_prompt-4", "evalita-mp_at_prompt-5", "evalita-mp_at_prompt-6", "evalita-mp_at_tasks",
    "evalita-mp_faq", "evalita-mp_faq_prompt-1", "evalita-mp_faq_prompt-2", "evalita-mp_faq_prompt-3",
    "evalita-mp_faq_prompt-4", "evalita-mp_faq_prompt-5", "evalita-mp_faq_prompt-6", "evalita-mp_faq_tasks",
    "evalita-mp_gen", "evalita-mp_hs", "evalita-mp_hs_prompt-1", "evalita-mp_hs_prompt-2",
    "evalita-mp_hs_prompt-3", "evalita-mp_hs_prompt-4", "evalita-mp_hs_prompt-5", "evalita-mp_hs_prompt-6",
    "evalita-mp_hs_tasks", "evalita-mp_ls", "evalita-mp_ls_prompt-1", "evalita-mp_ls_prompt-2",
    "evalita-mp_ls_tasks", "evalita-mp_mc", "evalita-mp_ner-v2_adg_p1", "evalita-mp_ner-v2_adg_p2",
    "evalita-mp_ner-v2_fic_p1", "evalita-mp_ner-v2_fic_p2", "evalita-mp_ner-v2_tasks_adg",
    "evalita-mp_ner-v2_tasks_fic", "evalita-mp_ner-v2_tasks_wn", "evalita-mp_ner-v2_wn_p1",
    "evalita-mp_ner-v2_wn_p2", "evalita-mp_ner_adg_group", "evalita-mp_ner_adg_p1", "evalita-mp_ner_adg_p2",
    "evalita-mp_ner_fic_group", "evalita-mp_ner_fic_p1", "evalita-mp_ner_fic_p2", "evalita-mp_ner_group",
    "evalita-mp_ner_tasks_adg", "evalita-mp_ner_tasks_fic", "evalita-mp_ner_tasks_wn",
    "evalita-mp_ner_wn_group", "evalita-mp_ner_wn_p1", "evalita-mp_ner_wn_p2", "evalita-mp_re",
    "evalita-mp_re_prompt-1", "evalita-mp_re_prompt-2", "evalita-mp_re_tasks", "evalita-mp_sa",
    "evalita-mp_sa_prompt-1", "evalita-mp_sa_prompt-2", "evalita-mp_sa_prompt-3", "evalita-mp_sa_prompt-4",
    "evalita-mp_sa_prompt-5", "evalita-mp_sa_prompt-6", "evalita-mp_sa_tasks", "evalita-mp_sum_fp",
    "evalita-mp_sum_fp-small_tasks", "evalita-mp_sum_fp_tasks", "evalita-mp_te", "evalita-mp_te_prompt-1",
    "evalita-mp_te_prompt-2", "evalita-mp_te_prompt-3", "evalita-mp_te_prompt-4", "evalita-mp_te_prompt-5",
    "evalita-mp_te_prompt-6", "evalita-mp_te_tasks", "evalita-mp_wic", "evalita-mp_wic_prompt-1",
    "evalita-mp_wic_prompt-2", "evalita-mp_wic_prompt-3", "evalita-mp_wic_prompt-4", "evalita-mp_wic_prompt-5",
    "evalita-mp_wic_prompt-6", "evalita-mp_wic_tasks",
)

class EvalitaMpExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Evalita Mp benchmark - Italian language medical, legal, and NLP tasks.

    This is a multiple choice task testing knowledge across various domains.
    Format: Question + A/B/C/D/E choices + Correct (letter A-E)
    """


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
        log = bind(_LOG, doc_id=doc.get("Id", doc.get("id", "unknown")))

        try:
            # Format 1: Textual Entailment format (text1, text2, entailment)
            if "text1" in doc and "text2" in doc and "entailment" in doc:
                text1 = str(doc.get("text1", "")).strip()
                text2 = str(doc.get("text2", "")).strip()
                entailment = str(doc.get("entailment", "")).strip().upper()

                if not text1 or not text2:
                    log.debug("Skipping doc due to missing text1 or text2", extra={"doc": doc})
                    return None

                # Format question
                formatted_question = f"Text 1: {text1}\nText 2: {text2}\nDoes text 2 entail text 1?"

                # Binary classification: entailment is either YES/SI or NO
                if entailment in ("YES", "SI", "SÌ"):
                    correct = "YES"
                    incorrect = "NO"
                elif entailment in ("NO"):
                    correct = "NO"
                    incorrect = "YES"
                else:
                    log.debug(f"Skipping doc due to unknown entailment value: {entailment}", extra={"doc": doc})
                    return None

                metadata = {"label": "evalita-mp"}
                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 2: Evalita format: Question (capital Q) + A/B/C/D/E + Correct (capital C)
            question = doc.get("Question", doc.get("question", "")).strip()

            # Build choices from A/B/C/D/E keys
            choices = []
            for letter in ["A", "B", "C", "D", "E"]:
                if letter in doc:
                    choices.append(str(doc[letter]).strip())

            # Get correct answer
            answer = doc.get("Correct", doc.get("answer", None))

            if not question or not choices:
                log.debug("Skipping doc due to missing question or choices", extra={"doc": doc})
                return None

            # Convert answer letter to index
            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                log.debug("Skipping doc due to invalid answer format", extra={"doc": doc})
                return None

            if not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to answer index out of range", extra={"doc": doc})
                return None

            # Get correct and incorrect answers
            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            # Format question with all choices
            choice_labels = ["A", "B", "C", "D", "E"][:len(choices)]
            formatted_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
            formatted_question = f"Question: {question}\n{formatted_choices}"

            metadata = {"label": "evalita-mp"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None



