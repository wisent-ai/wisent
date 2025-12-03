from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["BasqueBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "arc_eu_easy",
    "arc_eu_challenge",
    "paws_eu",
    "piqa_eu",
    "wnli_eu",
    "xcopa_eu",
    "mgsm_direct_eu",
)
class BasqueBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Basque Bench multiple-choice benchmarks."""


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
            log.warning("No valid Basque Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: ARC-eu (question, choices, answerKey)
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices = doc.get("choices", {})
                answer_key = doc.get("answerKey", "")

                if not question or not choices or not answer_key:
                    log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                    return None

                choice_labels = choices.get("label", [])
                choice_texts = choices.get("text", [])

                if not choice_labels or not choice_texts:
                    log.debug("Skipping doc due to missing choice data", extra={"doc": doc})
                    return None

                try:
                    correct_idx = choice_labels.index(answer_key)
                    correct = str(choice_texts[correct_idx]).strip()
                except (ValueError, IndexError):
                    log.debug("Invalid answer key", extra={"doc": doc})
                    return None

                incorrect_answers = [text for i, text in enumerate(choice_texts) if i != correct_idx]
                if not incorrect_answers:
                    return None

                incorrect = str(incorrect_answers[0]).strip()

                formatted_question = f"Galdera: {question}\nErantzuna:"

                positive_response = PositiveResponse(model_response=correct)
                negative_response = NegativeResponse(model_response=incorrect)

                return ContrastivePair(
                    prompt=formatted_question,
                    positive_response=positive_response,
                    negative_response=negative_response,
                    label="basque_bench_mc",
                )

            # Format 2: PAWS-eu (sentence1, sentence2, label)
            elif "sentence1" in doc and "sentence2" in doc and "label" in doc:
                sentence1 = str(doc.get("sentence1", "")).strip()
                sentence2 = str(doc.get("sentence2", "")).strip()
                label = doc.get("label")

                if not sentence1 or not sentence2:
                    return None

                # label: 1 = paraphrase, 0 = not paraphrase
                if label == 1:
                    correct = f"{sentence1}, ezta? Bai, {sentence2}"
                    incorrect = f"{sentence1}, ezta? Ez, {sentence2}"
                else:
                    correct = f"{sentence1}, ezta? Ez, {sentence2}"
                    incorrect = f"{sentence1}, ezta? Bai, {sentence2}"

                formatted_question = f"{sentence1}"

                positive_response = PositiveResponse(model_response=correct)
                negative_response = NegativeResponse(model_response=incorrect)

                return ContrastivePair(
                    prompt=formatted_question,
                    positive_response=positive_response,
                    negative_response=negative_response,
                    label="basque_bench_paws",
                )

            else:
                log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
                return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
