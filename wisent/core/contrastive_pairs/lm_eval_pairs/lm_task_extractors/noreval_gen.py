from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["NorevalGenerationExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    # ask_gec
    "ask_gec_p0", "ask_gec_p1", "ask_gec_p2", "ask_gec_p3", "ask_gec_p4",
    # norrewrite/norsummarize instruct
    "norrewrite_instruct",
    "norsummarize_instruct",
    # norsumm
    "norsumm_nno_p0", "norsumm_nno_p1", "norsumm_nno_p2", "norsumm_nno_p3", "norsumm_nno_p4", "norsumm_nno_p5",
    "norsumm_nob_p0", "norsumm_nob_p1", "norsumm_nob_p2", "norsumm_nob_p3", "norsumm_nob_p4", "norsumm_nob_p5",
    # nortruthfulqa_gen
    "nortruthfulqa_gen_nno_p0", "nortruthfulqa_gen_nno_p1", "nortruthfulqa_gen_nno_p2", "nortruthfulqa_gen_nno_p3", "nortruthfulqa_gen_nno_p4",
    "nortruthfulqa_gen_nob_p0", "nortruthfulqa_gen_nob_p1", "nortruthfulqa_gen_nob_p2", "nortruthfulqa_gen_nob_p3", "nortruthfulqa_gen_nob_p4",
    # tatoeba
    "tatoeba_eng_nno_p0", "tatoeba_eng_nno_p1", "tatoeba_eng_nno_p2", "tatoeba_eng_nno_p3",
    "tatoeba_eng_nob_p0", "tatoeba_eng_nob_p1", "tatoeba_eng_nob_p2", "tatoeba_eng_nob_p3",
    "tatoeba_nno_eng_p0", "tatoeba_nno_eng_p1", "tatoeba_nno_eng_p2", "tatoeba_nno_eng_p3",
    "tatoeba_nob_eng_p0", "tatoeba_nob_eng_p1", "tatoeba_nob_eng_p2", "tatoeba_nob_eng_p3",
)
class NorevalGenerationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Noreval generation benchmarks.

    Handles multiple Norwegian generation formats:
    1. ask_gec: Grammar error correction {source, correction}
    2. noridiom: Idiom completion {idiom_start, accepted_completions}
    3. norquad: QA {context, question, answers}
    4. norrewrite_instruct/norsummarize_instruct: Text transformation {prompt, context, target}
    5. nortruthfulqa_gen: TruthfulQA generation {question, correct_answers, incorrect_answers}

    For generation tasks, creates synthetic negative responses by:
    - Using source text as negative for correction tasks
    - Using wrong completions or shuffled text for other tasks
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Noreval generation docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Noreval.
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
            log.warning("No valid Noreval generation pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Noreval generation doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: ask_gec - {source, correction}
            if "source" in doc and "correction" in doc:
                source = str(doc["source"]).strip()
                correction = str(doc["correction"]).strip()

                if not source or not correction:
                    log.debug("Skipping doc due to empty source/correction", extra={"doc": doc})
                    return None

                question = f"Correct the following Norwegian sentence:\n{source}"
                correct = correction
                incorrect = source  # Use the uncorrected version as negative

                metadata = {"label": "noreval_gec"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 2: noridiom - {idiom_start, accepted_completions}
            elif "idiom_start" in doc and "accepted_completions" in doc:
                idiom_start = str(doc["idiom_start"]).strip()
                accepted_completions = doc["accepted_completions"]

                if not idiom_start or not accepted_completions:
                    log.debug("Skipping doc due to empty idiom fields", extra={"doc": doc})
                    return None

                correct = str(accepted_completions[0]).strip()

                # Create synthetic negative by using a different completion
                incorrect = "ukjent"  # "unknown" in Norwegian

                question = f"Complete the Norwegian idiom: {idiom_start}"

                metadata = {"label": "noreval_idiom"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 3: norquad - {context, question, answers}
            elif "context" in doc and "question" in doc and "answers" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc["question"]).strip()
                answers = doc["answers"]

                if not question or not answers:
                    log.debug("Skipping doc due to empty question/answers", extra={"doc": doc})
                    return None

                # Extract answer text
                if isinstance(answers, dict) and "text" in answers:
                    answer_texts = answers["text"]
                    if answer_texts:
                        correct = str(answer_texts[0]).strip()
                    else:
                        return None
                else:
                    return None

                # Create synthetic negative by shuffling words in the correct answer
                incorrect = self._create_shuffled_text(correct)

                formatted_question = f"Context: {context[:200]}...\n\nQuestion: {question}" if len(context) > 200 else f"Context: {context}\n\nQuestion: {question}"

                metadata = {"label": "noreval_qa"}

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 4: norrewrite_instruct/norsummarize_instruct - {prompt, context, target}
            elif "prompt" in doc and "context" in doc and "target" in doc:
                prompt = str(doc["prompt"]).strip()
                context = str(doc["context"]).strip()
                target = str(doc["target"]).strip()

                if not prompt or not context or not target:
                    log.debug("Skipping doc due to empty prompt/context/target", extra={"doc": doc})
                    return None

                question = f"{prompt}\n\n{context[:300]}..." if len(context) > 300 else f"{prompt}\n\n{context}"
                correct = target

                # Create synthetic negative by shuffling sentences in the target
                incorrect = self._create_shuffled_text(target)

                metadata = {"label": "noreval_rewrite"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 5: nortruthfulqa_gen - {question, correct_answers, incorrect_answers}
            elif "question" in doc and "correct_answers" in doc:
                question = str(doc["question"]).strip()
                correct_answers = doc.get("correct_answers", [])
                incorrect_answers = doc.get("incorrect_answers", [])

                if not question or not correct_answers:
                    log.debug("Skipping doc due to empty question/correct_answers", extra={"doc": doc})
                    return None

                correct = str(correct_answers[0]).strip()

                # Use provided incorrect answer if available, otherwise create synthetic
                if incorrect_answers:
                    incorrect = str(incorrect_answers[0]).strip()
                else:
                    incorrect = self._create_shuffled_text(correct)

                formatted_question = f"Question: {question}"

                metadata = {"label": "noreval_truthfulqa_gen"}

                return self._build_pair(
                    question=formatted_question,
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
    def _create_shuffled_text(text: str) -> str:
        """Create a synthetic negative response by shuffling words or sentences."""
        # Try to shuffle sentences first
        sentences = text.split(". ")
        if len(sentences) > 2:
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            return ". ".join(shuffled)

        # If only one sentence, shuffle words
        words = text.split()
        if len(words) > 3:
            shuffled = words.copy()
            random.shuffle(shuffled)
            return " ".join(shuffled)

        # If too short, just return a generic wrong answer
        return "feil svar"  # "wrong answer" in Norwegian

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
