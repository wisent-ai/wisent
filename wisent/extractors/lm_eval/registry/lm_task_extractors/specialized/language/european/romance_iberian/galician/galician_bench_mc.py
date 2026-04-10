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
            # MGSM schema: question + answer_number (math word problem)
            if "question" in doc and "answer_number" in doc and "mc1_targets" not in doc:
                q = str(doc.get("question", "")).strip()
                ans = str(doc.get("answer_number", "")).strip()
                if q and ans:
                    return ContrastivePair(
                        prompt=q + "\nResposta:",
                        positive_response=PositiveResponse(model_response=ans),
                        negative_response=NegativeResponse(model_response=str(int(float(ans)) + 1) if ans.replace(".","",1).isdigit() else "incorrecto"),
                        label="mgsm_direct_gl",
                    )

            # truthfulqa-multi format: question + mc1_targets dict
            if "question" in doc and "mc1_targets" in doc:
                question = str(doc.get("question", "")).strip()
                mc1 = doc.get("mc1_targets", {})
                if question and isinstance(mc1, dict):
                    choices_list = mc1.get("choices", [])
                    labels = mc1.get("labels", [])
                    if choices_list and labels and len(choices_list) == len(labels):
                        try:
                            correct_idx = labels.index(1)
                            correct = str(choices_list[correct_idx]).strip()
                            other = [c for i, c in enumerate(choices_list) if i != correct_idx]
                            if other:
                                return ContrastivePair(
                                    prompt=question,
                                    positive_response=PositiveResponse(model_response=correct),
                                    negative_response=NegativeResponse(model_response=str(other[0]).strip()),
                                    label="gl_bench_mc",
                                )
                        except (ValueError, IndexError):
                            pass

            # paws_gl format: sentence1 + sentence2 + label (binary paraphrase)
            if "sentence1" in doc and "sentence2" in doc and "label" in doc:
                s1 = str(doc.get("sentence1", "")).strip()
                s2 = str(doc.get("sentence2", "")).strip()
                try:
                    label_idx = int(doc.get("label", 0))
                except (TypeError, ValueError):
                    return None
                if not s1 or not s2:
                    return None
                prompt = f"{s1}, verdadeiro?"
                correct = "Si, " + s2 if label_idx == 1 else "Non, " + s2
                incorrect = "Non, " + s2 if label_idx == 1 else "Si, " + s2
                return ContrastivePair(
                    prompt=prompt,
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="paws_gl",
                )

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
