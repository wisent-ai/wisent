from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GalicianBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "belebele_glg_Latn",
    "galcola",
    "mgsm_direct_gl",
    "openbookqa_gl",
    "parafrases_gl",
    "paws_gl",
    "truthfulqa_gl_mc1",
    "truthfulqa_gl_mc2",
    "xnli_gl",
    "xstorycloze_gl"
)
class GalicianBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Galician Bench multiple-choice benchmarks."""


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
            log.warning("No valid Galician Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # parafrases_gl format: Frase + Paráfrase + Avaliación
            if "Frase" in doc and "Paráfrase" in doc and "Avaliación" in doc:
                frase = str(doc.get("Frase", "")).strip()
                parafrase = str(doc.get("Paráfrase", "")).strip()
                avaliacion = doc.get("Avaliación", -1)
                if frase and parafrase:
                    # Avaliación typically: 0-3 score; >=2 = paraphrase
                    is_paraphrase = float(avaliacion) >= 2.0 if avaliacion is not None else False
                    correct = "Si" if is_paraphrase else "No"
                    incorrect = "No" if is_paraphrase else "Si"
                    return ContrastivePair(
                        prompt=f"Frase 1: {frase}\nFrase 2: {parafrase}\nSon paráfrases?",
                        positive_response=PositiveResponse(model_response=correct),
                        negative_response=NegativeResponse(model_response=incorrect),
                        label="gl_bench_mc",
                    )

            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices = doc.get("choices", {})
                answer_key = doc.get("answerKey", "") or doc.get("answer", "")

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

                formatted_question = f"Question: {question}\nAnswer:"

                positive_response = PositiveResponse(model_response=correct)
                negative_response = NegativeResponse(model_response=incorrect)

                return ContrastivePair(
                    prompt=formatted_question,
                    positive_response=positive_response,
                    negative_response=negative_response,
                    label="gl_bench_mc",
                )

            else:
                log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
                return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
