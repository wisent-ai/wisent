from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["JapaneseLeaderboardGenerationExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "ja_leaderboard_jaqket_v2",
    "ja_leaderboard_jsquad",
    "ja_leaderboard_mgsm",
    "ja_leaderboard_xlsum",
)
class JapaneseLeaderboardGenerationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Japanese Leaderboard generation benchmarks."""


    evaluator_name = "generation"
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
            log.warning("No valid Japanese Leaderboard generation pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: JAQKET-v2 - {ctxs, question, answers}
            if "ctxs" in doc and "question" in doc and "answers" in doc:
                ctxs = doc.get("ctxs", [])
                question = str(doc.get("question", "")).strip()
                answers = doc.get("answers", {})

                if not question or not answers:
                    log.debug("Skipping doc due to missing question/answers", extra={"doc": doc})
                    return None

                # Extract context text
                context_text = ""
                if isinstance(ctxs, list) and ctxs:
                    context_texts = [str(ctx.get("text", "")) for ctx in ctxs if isinstance(ctx, dict)]
                    context_text = " ".join(context_texts)

                # Extract answer text
                if isinstance(answers, dict) and "text" in answers:
                    answer_texts = answers["text"]
                    if answer_texts:
                        correct = str(answer_texts[0]).strip()
                    else:
                        return None
                else:
                    return None

                # Create synthetic negative by shuffling words
                incorrect = self._create_shuffled_text(correct)

                formatted_question = f"Context: {context_text[:200]}...\n\nQuestion: {question}" if len(context_text) > 200 else f"Context: {context_text}\n\nQuestion: {question}"

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "japanese_leaderboard_jaqket"},
                )

            # Format 2: JSQuAD - {context, question, answers}
            elif "context" in doc and "question" in doc and "answers" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answers = doc.get("answers", {})

                if not question or not answers:
                    log.debug("Skipping doc due to missing question/answers", extra={"doc": doc})
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

                # Create synthetic negative by shuffling words
                incorrect = self._create_shuffled_text(correct)

                # Extract context after [SEP] if present
                if "[SEP]" in context:
                    context = context.split("[SEP]")[-1].strip()

                formatted_question = f"Context: {context[:200]}...\n\nQuestion: {question}" if len(context) > 200 else f"Context: {context}\n\nQuestion: {question}"

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "japanese_leaderboard_jsquad"},
                )

            # Format 3: MGSM - {question, answer}
            elif "question" in doc and "answer" in doc:
                question = str(doc.get("question", "")).strip()
                answer = str(doc.get("answer", "")).strip()

                if not question or not answer:
                    log.debug("Skipping doc due to missing question/answer", extra={"doc": doc})
                    return None

                # Remove prefixes
                question = question.replace("問題：", "").strip()
                correct = answer.replace("ステップごとの答え：", "").strip()

                # Create synthetic negative by shuffling
                incorrect = self._create_shuffled_text(correct)

                formatted_question = f"Question: {question}"

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "japanese_leaderboard_mgsm"},
                )

            # Format 4: XL-Sum - {text, summary}
            elif "text" in doc and "summary" in doc:
                text = str(doc.get("text", "")).strip()
                summary = str(doc.get("summary", "")).strip()

                if not text or not summary:
                    log.debug("Skipping doc due to missing text/summary", extra={"doc": doc})
                    return None

                correct = summary

                # Create synthetic negative by shuffling
                incorrect = self._create_shuffled_text(correct)

                formatted_question = f"Summarize the following text:\n\n{text[:300]}..." if len(text) > 300 else f"Summarize the following text:\n\n{text}"

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "japanese_leaderboard_xlsum"},
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
        sentences = text.split("。")
        if len(sentences) > 2:
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            return "。".join(shuffled)

        # If only one sentence, shuffle words
        words = text.split()
        if len(words) > 3:
            shuffled = words.copy()
            random.shuffle(shuffled)
            return " ".join(shuffled)

        # If too short, just return a generic wrong answer
        return "間違った答え"

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
