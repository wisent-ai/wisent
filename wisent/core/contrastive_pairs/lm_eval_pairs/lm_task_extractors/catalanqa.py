from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CatalanqaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("catalanqa",)

class CatalanqaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Catalanqa benchmark (SQuAD-like QA)."""


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
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: SQuAD-like format with context + question + answers
            if "context" in doc and "question" in doc and "answers" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answers_data = doc.get("answers", [])

                if not context or not question or not answers_data:
                    log.debug("Skipping doc - missing context/question/answers", extra={"doc": doc})
                    return None

                # Extract correct answer text
                if isinstance(answers_data, list) and len(answers_data) > 0:
                    correct_answer = str(answers_data[0].get("text", "")).strip()
                elif isinstance(answers_data, dict):
                    texts = answers_data.get("text", [])
                    if texts and len(texts) > 0:
                        correct_answer = str(texts[0]).strip()
                    else:
                        log.debug("Skipping doc - no answer text", extra={"doc": doc})
                        return None
                else:
                    log.debug("Skipping doc - invalid answers format", extra={"doc": doc})
                    return None

                if not correct_answer:
                    log.debug("Skipping doc - empty correct answer", extra={"doc": doc})
                    return None

                # Create synthetic negative by extracting a different span from context
                # Split context into sentences/phrases and pick one that's not the answer
                import re
                sentences = [s.strip() for s in re.split(r'[.!?]\s+', context) if s.strip()]

                incorrect_answer = None
                for sent in sentences:
                    # Extract a phrase from the sentence
                    words = sent.split()
                    if len(words) >= 2:
                        # Take first 2-4 words as potential incorrect answer
                        phrase = ' '.join(words[:min(4, len(words))])
                        if phrase.lower() != correct_answer.lower() and len(phrase) > 3:
                            incorrect_answer = phrase
                            break

                if not incorrect_answer:
                    # Fallback: use first few words of context
                    words = context.split()[:3]
                    incorrect_answer = ' '.join(words) if words else "incorrect answer"

                # Format prompt
                prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

                metadata = {"label": "catalanqa"}

                return self._build_pair(
                    question=prompt,
                    correct=correct_answer,
                    incorrect=incorrect_answer,
                    metadata=metadata,
                )

            # Format 2: Multiple choice fallback
            question = doc.get("question", doc.get("query", doc.get("input", doc.get("instruction", doc.get("prompt", ""))))).strip()
            
            # Try multiple format patterns for choices
            choices = doc.get("choices", doc.get("options", doc.get("answers", [])))
            
            # Handle option_a/b/c/d format
            if not choices and "option_a" in doc:
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]

            # Try multiple format patterns for answer
            answer = doc.get("answer", doc.get("label", doc.get("target", None)))

            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                return None

            if not question or not choices or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()
            metadata = {"label": "catalanqa"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

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
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
