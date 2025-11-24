from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["BasqueglueExtractor"]
_LOG = setup_logger(__name__)

task_names = ("basque-glue", "basqueglue")

evaluator_name = "log_likelihoods"


class BasqueglueExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for the Basqueglue benchmark."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="basqueglue")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("basqueglue", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load basqueglue dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(dataset)})

        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "basqueglue"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("answer", doc.get("answerKey", ""))
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    answer_idx = int(answer) if answer else 0

            # Format 2: instruction + option_a/b/c/d + answer (MMMLU style)
            elif "instruction" in doc and "option_a" in doc:
                question = str(doc.get("instruction", "")).strip()
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')

            # Format 3: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "basqueglue"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            formatted_question = f"Question: {question}\nA. {incorrect}\nB. {correct}"

            metadata = {"label": "basqueglue"}

            return self._build_pair(
                question=formatted_question,
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
