from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["NorevalMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    # ncb
    "ncb",
    # norbelebele
    "norbelebele",
    "norbelebele_p0", "norbelebele_p1", "norbelebele_p2", "norbelebele_p3", "norbelebele_p4",
    # norcommonsenseqa
    "norcommonsenseqa_nno", "norcommonsenseqa_nob",
    "norcommonsenseqa_nno_p0", "norcommonsenseqa_nno_p1", "norcommonsenseqa_nno_p2", "norcommonsenseqa_nno_p3", "norcommonsenseqa_nno_p4",
    "norcommonsenseqa_nob_p0", "norcommonsenseqa_nob_p1", "norcommonsenseqa_nob_p2", "norcommonsenseqa_nob_p3", "norcommonsenseqa_nob_p4",
    # norec
    "norec_document_p0", "norec_document_p1", "norec_document_p2", "norec_document_p3", "norec_document_p4",
    "norec_sentence_p0", "norec_sentence_p1", "norec_sentence_p2", "norec_sentence_p3", "norec_sentence_p4",
    # noropenbookqa
    "noropenbookqa_nno", "noropenbookqa_nob",
    "noropenbookqa_nno_p0", "noropenbookqa_nno_p1", "noropenbookqa_nno_p2", "noropenbookqa_nno_p3", "noropenbookqa_nno_p4",
    "noropenbookqa_nob_p0", "noropenbookqa_nob_p1", "noropenbookqa_nob_p2", "noropenbookqa_nob_p3", "noropenbookqa_nob_p4",
    # nortruthfulqa_mc
    "nortruthfulqa_mc_nno", "nortruthfulqa_mc_nob",
    "nortruthfulqa_mc_nno_p0", "nortruthfulqa_mc_nno_p1", "nortruthfulqa_mc_nno_p2", "nortruthfulqa_mc_nno_p3", "nortruthfulqa_mc_nno_p4",
    "nortruthfulqa_mc_nob_p0", "nortruthfulqa_mc_nob_p1", "nortruthfulqa_mc_nob_p2", "nortruthfulqa_mc_nob_p3", "nortruthfulqa_mc_nob_p4",
    # nrk_quiz_qa
    "nrk_quiz_qa_nno", "nrk_quiz_qa_nob",
    "nrk_quiz_qa_nno_p0", "nrk_quiz_qa_nno_p1", "nrk_quiz_qa_nno_p2", "nrk_quiz_qa_nno_p3", "nrk_quiz_qa_nno_p4",
    "nrk_quiz_qa_nob_p0", "nrk_quiz_qa_nob_p1", "nrk_quiz_qa_nob_p2", "nrk_quiz_qa_nob_p3", "nrk_quiz_qa_nob_p4",
)
class NorevalMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Noreval multiple-choice benchmarks.

    Handles three different Norwegian multiple-choice formats:
    1. NCB: Simple correct/wrong pairs for grammar correction
    2. NorTruthfulQA: TruthfulQA-style with question + mc1_targets {choices, labels}
    3. NRK Quiz QA: Standard multiple-choice with question + choices + answer
    """


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Noreval docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Noreval.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
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
            log.warning("No valid Noreval pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Noreval doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # NoREC document sentiment: review + sentiment (binary 0/1)
            if "review" in doc and "sentiment" in doc:
                review = str(doc.get("review", "")).strip()
                sentiment = doc.get("sentiment")
                if review and sentiment in (0, 1):
                    correct = "Positivt" if sentiment == 1 else "Negativt"
                    incorrect = "Negativt" if sentiment == 1 else "Positivt"
                    return self._build_pair(
                        question=f"Tekst: {review[:1500]}\nSentiment:",
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "noreval_norec"},
                    )

            # NorBeleBele: question + flores_passage + mc_answer1-4 + correct_answer_num
            if "flores_passage" in doc and "mc_answer1" in doc and "correct_answer_num" in doc:
                passage = str(doc.get("flores_passage", "")).strip()
                question = str(doc.get("question", "")).strip()
                answers = [str(doc.get(f"mc_answer{i}", "")).strip() for i in range(1, 5)]
                correct_num = doc.get("correct_answer_num")
                try:
                    correct_idx = int(correct_num) - 1
                except (TypeError, ValueError):
                    return None
                if passage and question and answers and 0 <= correct_idx < len(answers):
                    return self._build_pair(
                        question=f"Passasje: {passage}\nSpørsmål: {question}",
                        correct=answers[correct_idx],
                        incorrect=answers[(correct_idx + 1) % len(answers)],
                        metadata={"label": "noreval_belebele"},
                    )

            # NorOpenBookQA: question_stem + choices + answer/answerKey
            if "question_stem" in doc and "choices" in doc:
                question = str(doc.get("question_stem", "")).strip()
                choices = doc.get("choices", {})
                answer_key = doc.get("answerKey", "") or doc.get("answer", "")
                if question and isinstance(choices, dict) and answer_key:
                    choice_texts = choices.get("text", [])
                    choice_labels = choices.get("label", [])
                    if choice_texts and choice_labels:
                        try:
                            correct_idx = choice_labels.index(answer_key)
                            return self._build_pair(
                                question=f"Spørsmål: {question}",
                                correct=str(choice_texts[correct_idx]).strip(),
                                incorrect=str(choice_texts[(correct_idx + 1) % len(choice_texts)]).strip(),
                                metadata={"label": "noreval_obqa"},
                            )
                        except (ValueError, IndexError):
                            pass

            # NorRewrite/NorSummarize prompt-context-response format
            if "prompt" in doc and "context" in doc and "response" in doc:
                prompt = str(doc.get("prompt", "")).strip()
                context = str(doc.get("context", "")).strip()
                response = str(doc.get("response", "")).strip()
                if prompt and response:
                    full_prompt = f"{prompt}\n{context}".strip()
                    words = response.split()
                    incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil svar"
                    return self._build_pair(
                        question=full_prompt,
                        correct=response,
                        incorrect=incorrect,
                        metadata={"label": "noreval_instruct"},
                    )

            # NorSumm: article + summaries (list)
            if "article" in doc and "summaries" in doc:
                article = str(doc.get("article", "")).strip()
                summaries = doc.get("summaries", [])
                if article and summaries:
                    if isinstance(summaries, list) and summaries:
                        summary = str(summaries[0]).strip() if not isinstance(summaries[0], dict) else str(summaries[0].get("text", summaries[0].get("summary", ""))).strip()
                    else:
                        summary = str(summaries).strip()
                    if summary:
                        words = summary.split()
                        incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil samandrag"
                        return self._build_pair(
                            question=f"Sammendrag av: {article[:1500]}",
                            correct=summary,
                            incorrect=incorrect,
                            metadata={"label": "noreval_summ"},
                        )

            # Tatoeba: sourceString + targetString
            if "sourceString" in doc and "targetString" in doc:
                src = str(doc.get("sourceString", "")).strip()
                tgt = str(doc.get("targetString", "")).strip()
                if src and tgt:
                    words = tgt.split()
                    incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil"
                    return self._build_pair(
                        question=f"Oversett: {src}",
                        correct=tgt,
                        incorrect=incorrect,
                        metadata={"label": "noreval_tatoeba"},
                    )

            # NorRewrite/NorSummarize: instruction + (input or text) + output
            if ("instruction" in doc or "prompt" in doc) and ("output" in doc or "target" in doc or "summary" in doc):
                instr = str(doc.get("instruction", doc.get("prompt", ""))).strip()
                inp = str(doc.get("input", doc.get("text", ""))).strip()
                target = str(doc.get("output", doc.get("target", doc.get("summary", "")))).strip()
                if instr and target:
                    prompt = f"{instr}\n{inp}".strip()
                    # Synthetic incorrect: reverse target words
                    words = target.split()
                    incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil svar"
                    return self._build_pair(
                        question=prompt,
                        correct=target,
                        incorrect=incorrect,
                        metadata={"label": "noreval_instruct"},
                    )

            # Tatoeba translation: source_text + target_text or sentence_a/sentence_b
            if "source_text" in doc and "target_text" in doc:
                src = str(doc.get("source_text", "")).strip()
                tgt = str(doc.get("target_text", "")).strip()
                if src and tgt:
                    words = tgt.split()
                    incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil"
                    return self._build_pair(
                        question=f"Oversett: {src}",
                        correct=tgt,
                        incorrect=incorrect,
                        metadata={"label": "noreval_tatoeba"},
                    )

            # Format 1: NCB - {correct, wrong}
            if "correct" in doc and "wrong" in doc:
                correct = str(doc["correct"]).strip()
                incorrect = str(doc["wrong"]).strip()

                if not correct or not incorrect:
                    log.debug("Skipping doc due to empty correct/wrong fields", extra={"doc": doc})
                    return None

                prompt = "Which sentence is grammatically correct?"

                metadata = {"label": "noreval_ncb"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 2: NorTruthfulQA - {question, mc1_targets: {choices, labels}}
            elif "question" in doc and "mc1_targets" in doc:
                question = str(doc["question"]).strip()
                mc1_targets = doc["mc1_targets"]

                if not isinstance(mc1_targets, dict):
                    log.debug("Skipping doc due to invalid mc1_targets", extra={"doc": doc})
                    return None

                choices = mc1_targets.get("choices", [])
                labels = mc1_targets.get("labels", [])

                if not choices or not labels or len(choices) != len(labels):
                    log.debug("Skipping doc due to mismatched choices/labels", extra={"doc": doc})
                    return None

                # Find correct and incorrect answers
                correct_idx = None
                incorrect_idx = None

                for i, label in enumerate(labels):
                    if label == 1:
                        correct_idx = i
                    elif label == 0 and incorrect_idx is None:
                        incorrect_idx = i

                if correct_idx is None or incorrect_idx is None:
                    log.debug("Skipping doc due to missing correct/incorrect labels", extra={"doc": doc})
                    return None

                correct = str(choices[correct_idx]).strip()
                incorrect = str(choices[incorrect_idx]).strip()

                metadata = {"label": "noreval_truthfulqa"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 3: NRK Quiz QA - {question, choices: {label, text}, answer}
            elif "question" in doc and "choices" in doc and "answer" in doc:
                question = str(doc["question"]).strip()
                choices_data = doc["choices"]
                answer = str(doc["answer"]).strip()

                if not isinstance(choices_data, dict):
                    log.debug("Skipping doc due to invalid choices", extra={"doc": doc})
                    return None

                choice_texts = choices_data.get("text", [])
                choice_labels = choices_data.get("label", [])

                if not choice_texts or not choice_labels or len(choice_texts) != len(choice_labels):
                    log.debug("Skipping doc due to mismatched choice texts/labels", extra={"doc": doc})
                    return None

                # Find the correct answer index
                try:
                    answer_idx = choice_labels.index(answer)
                except ValueError:
                    log.debug("Skipping doc due to answer not in choices", extra={"doc": doc, "answer": answer})
                    return None

                correct = str(choice_texts[answer_idx]).strip()

                # Pick a different choice as incorrect
                incorrect_idx = (answer_idx + 1) % len(choice_texts)
                incorrect = str(choice_texts[incorrect_idx]).strip()

                metadata = {"label": "noreval_nrk_quiz"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            else:
                log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
                return None

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
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
        )
