from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CatalanBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "arc_ca_easy",
    "arc_ca_challenge",
    "catalanqa",
    "catcola",
    "cocoteros_va",
    "copa_ca",
    "coqcat",
    "mgsm_direct_ca",
    "openbookqa_ca",
    "parafraseja",
    "paws_ca"
)
class CatalanBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Catalan Bench multiple-choice benchmarks."""


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
            log.warning("No valid Catalan Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # MGSM schema: question + answer_number (math word problem)
            if "question" in doc and "answer_number" in doc and "premise" not in doc:
                q = str(doc.get("question", "")).strip()
                ans = str(doc.get("answer_number", "")).strip()
                if q and ans:
                    return ContrastivePair(
                        prompt=q + "\nResposta:",
                        positive_response=PositiveResponse(model_response=ans),
                        negative_response=NegativeResponse(model_response=str(int(float(ans)) + 1) if ans.replace(".","",1).isdigit() else "incorrect"),
                        label="mgsm_direct_ca",
                    )

            # COPA schema: premise + choice1 + choice2 + question + label
            if "premise" in doc and "choice1" in doc and "choice2" in doc and "label" in doc:
                premise = str(doc.get("premise", "")).strip()
                c1 = str(doc.get("choice1", "")).strip()
                c2 = str(doc.get("choice2", "")).strip()
                question_word = str(doc.get("question", "cause")).strip()
                label = int(doc.get("label", 0))
                if premise and c1 and c2 and 0 <= label <= 1:
                    correct = c1 if label == 0 else c2
                    incorrect = c2 if label == 0 else c1
                    connector = "perquè" if question_word == "cause" else "per tant"
                    return ContrastivePair(
                        prompt=f"{premise} {connector}",
                        positive_response=PositiveResponse(model_response=correct),
                        negative_response=NegativeResponse(model_response=incorrect),
                        label="ca_bench_mc",
                    )
                return None

            # Cocoteros generation schema: text + keywords + context
            if "text" in doc and "keywords" in doc and "context" in doc:
                text = str(doc.get("text", "")).strip()
                kw_raw = doc.get("keywords", "")
                if isinstance(kw_raw, list):
                    keywords = ", ".join(str(k) for k in kw_raw)
                else:
                    keywords = str(kw_raw).strip()
                context = str(doc.get("context", "")).strip()
                if text and keywords:
                    prompt = f"Genera una frase amb aquestes paraules: {keywords}. El context és: {context}\n\nResposta:"
                    # Synthetic incorrect: reverse word order
                    words = text.split()
                    incorrect_text = " ".join(reversed(words)) if len(words) > 1 else "frase incorrecta"
                    return ContrastivePair(
                        prompt=prompt,
                        positive_response=PositiveResponse(model_response=text),
                        negative_response=NegativeResponse(model_response=incorrect_text),
                        label="ca_bench_mc",
                    )
                return None

            # OpenBookQA-style: question + choices{label,text} + answerKey
            if ("question" in doc or "question_stem" in doc) and isinstance(doc.get("choices"), dict):
                question = str(doc.get("question", doc.get("question_stem", ""))).strip()
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
                    label="ca_bench_mc",
                )

            # paws_ca: sentence1 + sentence2 + label (binary paraphrase)
            if "sentence1" in doc and "sentence2" in doc and "label" in doc:
                s1 = str(doc.get("sentence1", "")).strip()
                s2 = str(doc.get("sentence2", "")).strip()
                try:
                    label_idx = int(doc.get("label", 0))
                except (TypeError, ValueError):
                    return None
                if not s1 or not s2:
                    return None
                prompt = f"{s1}, veritat?"
                correct = "Sí, " + s2 if label_idx == 1 else "No, " + s2
                incorrect = "No, " + s2 if label_idx == 1 else "Sí, " + s2
                return ContrastivePair(
                    prompt=prompt,
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="paws_ca",
                )

            # catcola: Sentence + Label (grammaticality binary)
            if "Sentence" in doc and "Label" in doc:
                sentence = str(doc.get("Sentence", "")).strip()
                try:
                    label_idx = int(doc.get("Label", 0))
                except (TypeError, ValueError):
                    return None
                if not sentence:
                    return None
                prompt = f"{sentence}\nPregunta: Té sentit aquesta frase?\nResposta:"
                correct = "sí" if label_idx == 1 else "no"
                incorrect = "no" if label_idx == 1 else "sí"
                return ContrastivePair(
                    prompt=prompt,
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="catcola",
                )

            # catalanqa / coqcat: context + question + answers (extractive QA)
            if "context" in doc and "question" in doc and "answers" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answers = doc.get("answers", {})
                # answers can be {"text": [...], "answer_start": [...]} or list of dicts
                if isinstance(answers, dict):
                    texts = answers.get("text", [])
                elif isinstance(answers, list) and answers and isinstance(answers[0], dict):
                    texts = [a.get("text", "") for a in answers]
                else:
                    return None
                if not texts:
                    return None
                correct = str(texts[0]).strip()
                if not correct or not context or not question:
                    return None
                prompt = f"Context: {context}\n\nPregunta: {question}\n\nResposta:"
                # Synthetic incorrect: reverse words
                words = correct.split()
                incorrect = " ".join(reversed(words)) if len(words) > 1 else "resposta incorrecta"
                return ContrastivePair(
                    prompt=prompt,
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="catalanqa",
                )

            log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
