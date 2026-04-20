from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FrenchBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "french_bench_arc_challenge",
    "french_bench_boolqa",
    "french_bench_hellaswag",
    "french_bench_multifquad",
    "french_bench_reading_comp",
    "french_bench_topic_based_nli",
    "french_bench_trivia",
    "french_bench_vocab",
    "french_bench_xnli"
)
class FrenchBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for French Bench multiple-choice benchmarks."""


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
            log.warning("No valid French Bench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # boolqa format: question + passage + label (binary)
            if "question" in doc and "passage" in doc and "label" in doc:
                question = str(doc.get("question", "")).strip()
                passage = str(doc.get("passage", "")).strip()
                label = doc.get("label", -1)
                if question and passage and label in (0, 1):
                    correct = "Oui" if label == 1 else "Non"
                    incorrect = "Non" if label == 1 else "Oui"
                    return ContrastivePair(
                        prompt=f"Passage: {passage}\nQuestion: {question}\nRéponse:",
                        positive_response=PositiveResponse(model_response=correct),
                        negative_response=NegativeResponse(model_response=incorrect),
                        label="french_bench_mc",
                    )

            # xnli: premise + hypothesis + label (0=entailment, 1=neutral, 2=contradiction)
            if "premise" in doc and "hypothesis" in doc and "label" in doc:
                premise = str(doc.get("premise", "")).strip()
                hypothesis = str(doc.get("hypothesis", "")).strip()
                label = doc.get("label")
                label_map = {0: "oui", 1: "peut-être", 2: "non"}
                try:
                    l_idx = int(label)
                except Exception:
                    l_idx = -1
                correct_nli = label_map.get(l_idx)
                if premise and hypothesis and correct_nli:
                    incorrect_nli = next(v for k, v in label_map.items() if v != correct_nli)
                    return ContrastivePair(
                        prompt=f"Prémisse: {premise}\nHypothèse: {hypothesis}\nLa prémisse implique-t-elle l'hypothèse?",
                        positive_response=PositiveResponse(model_response=correct_nli),
                        negative_response=NegativeResponse(model_response=incorrect_nli),
                        label="french_bench_xnli",
                    )

            # topic_based_nli: text + topic + polarity (positif/negatif/neutre)
            if "text" in doc and "topic" in doc and "polarity" in doc:
                text = str(doc.get("text", "")).strip()
                topic = str(doc.get("topic", "")).strip()
                polarity = str(doc.get("polarity", "")).strip().lower()
                polarity_choices = ["positif", "négatif", "neutre"]
                polarity_aliases = {"positif": "positif", "negatif": "négatif", "négatif": "négatif", "neutre": "neutre"}
                correct_choice = polarity_aliases.get(polarity)
                if not text or not topic or correct_choice is None:
                    return None
                incorrect = next(c for c in polarity_choices if c != correct_choice)
                return ContrastivePair(
                    prompt=f"\nAvis Client: {text}\n\nA propos du thème \"{topic}\", l'avis client est",
                    positive_response=PositiveResponse(model_response=correct_choice),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="french_bench_topic_based_nli",
                )

            # reading_comp: context + question + answerA/B/C/D + answer
            if "context" in doc and "question" in doc and "answerA" in doc and "answer" in doc:
                context = str(doc.get("context", "")).strip()
                q_text = str(doc.get("question", "")).strip()
                answers_letters = ["A", "B", "C", "D"]
                answers_texts = [str(doc.get(f"answer{l}", "")).strip() for l in answers_letters]
                answer_letter = str(doc.get("answer", "")).strip().upper()
                if answer_letter not in answers_letters:
                    return None
                correct_idx = answers_letters.index(answer_letter)
                correct = answers_texts[correct_idx]
                incorrect_candidates = [a for i, a in enumerate(answers_texts) if i != correct_idx and a]
                if not correct or not incorrect_candidates:
                    return None
                return ContrastivePair(
                    prompt=f"Context: {context}\n\n{q_text}",
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect_candidates[0]),
                    label="french_bench_reading_comp",
                )

            # trivia: Question + Answer (generation)
            if "Question" in doc and "Answer" in doc:
                q_text = str(doc.get("Question", "")).strip()
                a_text = str(doc.get("Answer", "")).strip()
                if not q_text or not a_text:
                    return None
                # Synthetic incorrect: reversed words
                words = a_text.split()
                incorrect = " ".join(reversed(words)) if len(words) > 1 else "réponse incorrecte"
                return ContrastivePair(
                    prompt=f"{q_text}\nAnswer:",
                    positive_response=PositiveResponse(model_response=a_text),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="french_bench_trivia",
                )

            # multifquad: context + question + answers (extractive QA)
            if "context" in doc and "question" in doc and "answers" in doc:
                context = str(doc.get("context", "")).strip()
                q_text = str(doc.get("question", "")).strip()
                answers = doc.get("answers", {})
                if isinstance(answers, dict):
                    texts = answers.get("text", [])
                elif isinstance(answers, list) and answers and isinstance(answers[0], dict):
                    texts = [a.get("text", "") for a in answers]
                else:
                    return None
                if not texts:
                    return None
                correct = str(texts[0]).strip()
                if not correct or not context or not q_text:
                    return None
                words = correct.split()
                incorrect = " ".join(reversed(words)) if len(words) > 1 else "réponse incorrecte"
                return ContrastivePair(
                    prompt=f"\nContexte: {context}\n\nQuestion: {q_text}\n\nRéponse:",
                    positive_response=PositiveResponse(model_response=correct),
                    negative_response=NegativeResponse(model_response=incorrect),
                    label="french_bench_multifquad",
                )

            # Format 1: Standard MC with question + choices + answerKey
            if "question" in doc and "choices" in doc and "answerKey" in doc:
                question = str(doc.get("question", "")).strip()
                choices = doc.get("choices", [])
                answer_key = doc.get("answerKey", "")

                if not question or not choices or not answer_key:
                    return None

                answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer_idx = answer_map.get(answer_key.upper())

                if answer_idx is None or answer_idx >= len(choices):
                    return None

            # Format 2: Hellaswag style with ctx + endings + label
            elif "ctx" in doc and "endings" in doc and "label" in doc:
                question = str(doc.get("ctx", "")).strip()
                choices = doc.get("endings", [])
                label = doc.get("label", "0")

                if not question or not choices:
                    return None

                answer_idx = int(label) if isinstance(label, (int, str)) and str(label).isdigit() else 0
                if answer_idx >= len(choices):
                    return None

            else:
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            formatted_question = f"Question: {question}\nRéponse:"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="french_bench_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
