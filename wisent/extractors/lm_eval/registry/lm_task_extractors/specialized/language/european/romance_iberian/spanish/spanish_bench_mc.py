from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SpanishBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "cocoteros_es",
    "copa_es",
    "escola",
    "mgsm_direct_es_spanish_bench",
    "openbookqa_es",
    "paws_es_spanish_bench",
    "wnli_es",
    "xnli_es_spanish_bench"
)
class SpanishBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Spanish Bench multiple-choice benchmarks."""


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
            log.warning("No valid Spanish Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # MGSM schema: question + answer_number (math word problem)
            if "question" in doc and "answer_number" in doc and "choices" not in doc:
                q = str(doc.get("question", "")).strip()
                ans = str(doc.get("answer_number", "")).strip()
                if q and ans:
                    return ContrastivePair(
                        prompt=q + "\nRespuesta:",
                        positive_response=PositiveResponse(model_response=ans),
                        negative_response=NegativeResponse(model_response=str(int(float(ans)) + 1) if ans.replace(".","",1).isdigit() else "incorrecto"),
                        label="mgsm_direct_es",
                    )

            # OpenBookQA-style: (question | question_stem) + choices{label,text} + answerKey
            if ("question" in doc or "question_stem" in doc) and "choices" in doc and isinstance(doc.get("choices"), dict):
                question = str(doc.get("question", doc.get("question_stem", ""))).strip()
                choices = doc.get("choices", {})
                answer_key = doc.get("answerKey", "") or doc.get("answer", "")
                if isinstance(answer_key, str):
                    answer_key = answer_key.strip()

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
                    label="es_bench_mc",
                )

            # COPA-es: premise + question(cause/effect) + choice1 + choice2 + label
            if "premise" in doc and "choice1" in doc and "choice2" in doc and "label" in doc:
                premise = str(doc.get("premise", "")).strip()
                q_type = str(doc.get("question", "")).strip()
                choice1 = str(doc.get("choice1", "")).strip()
                choice2 = str(doc.get("choice2", "")).strip()
                label = doc.get("label", 0)

                if not premise or not choice1 or not choice2:
                    return None

                connector = {"cause": "porque", "effect": "y por lo tanto"}.get(q_type, "")
                prompt_text = f"{premise.rstrip('.')} {connector}".strip()

                try:
                    label_idx = int(label)
                except (TypeError, ValueError):
                    return None

                correct = choice1 if label_idx == 0 else choice2
                incorrect = choice2 if label_idx == 0 else choice1

                return ContrastivePair(
                    prompt=prompt_text,
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="copa_es",
                )

            # NLI-style: sentence1 + sentence2 + label (paws_es, wnli_es)
            if "sentence1" in doc and "sentence2" in doc and "label" in doc:
                s1 = str(doc.get("sentence1", "")).strip()
                s2 = str(doc.get("sentence2", "")).strip()
                label = doc.get("label", 0)
                try:
                    label_idx = int(label)
                except (TypeError, ValueError):
                    return None

                prompt_text = f"Frase 1: {s1}\nFrase 2: {s2}\n¿Son equivalentes?"
                # Binary entailment: 1 = equivalent, 0 = not equivalent
                positive_label = "Sí" if label_idx == 1 else "No"
                negative_label = "No" if label_idx == 1 else "Sí"

                return ContrastivePair(
                    prompt=prompt_text,
                    positive_response=PositiveResponse(model_response=positive_label),
                    negative_response=NegativeResponse(model_response=negative_label),
                    label="es_nli",
                )

            # XNLI-style: premise + hypothesis + label (3-class: entailment, neutral, contradiction)
            if "premise" in doc and "hypothesis" in doc and "label" in doc:
                premise = str(doc.get("premise", "")).strip()
                hypothesis = str(doc.get("hypothesis", "")).strip()
                label = doc.get("label", 0)
                try:
                    label_idx = int(label)
                except (TypeError, ValueError):
                    return None

                xnli_labels = ["Verdadero", "Ni verdadero ni falso", "Falso"]
                if not (0 <= label_idx < len(xnli_labels)):
                    return None
                correct = xnli_labels[label_idx]
                incorrect = xnli_labels[(label_idx + 1) % len(xnli_labels)]

                prompt_text = f"Premisa: {premise}\nHipótesis: {hypothesis}"

                return ContrastivePair(
                    prompt=prompt_text,
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="xnli_es",
                )

            log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
