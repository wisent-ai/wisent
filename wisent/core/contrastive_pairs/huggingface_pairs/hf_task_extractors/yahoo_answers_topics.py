from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["YahooAnswersTopicsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("yahoo_answers_topics",)

evaluator_name = "log_likelihoods"


class YahooAnswersTopicsExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Yahoo Answers Topics - topic classification."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="yahoo_answers_topics")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("yahoo_answers_topics", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load yahoo_answers_topics dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "yahoo_answers_topics"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question_title = doc.get("question_title", "").strip()
            question_content = doc.get("question_content", "").strip()
            best_answer = doc.get("best_answer", "").strip()
            topic = doc.get("topic", doc.get("label", None))

            if not question_title or topic is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Yahoo Answers topics: 10 categories
            topics = [
                "Society & Culture",
                "Science & Mathematics",
                "Health",
                "Education & Reference",
                "Computers & Internet",
                "Sports",
                "Business & Finance",
                "Entertainment & Music",
                "Family & Relationships",
                "Politics & Government"
            ]

            if not isinstance(topic, int) or not (0 <= topic < len(topics)):
                log.debug("Invalid topic", extra={"doc": doc})
                return None

            correct = topics[topic]
            incorrect_idx = (topic + 1) % len(topics)
            incorrect = topics[incorrect_idx]

            # Combine question parts
            full_question = question_title
            if question_content:
                full_question += f" {question_content}"

            prompt = f"What topic does this question belong to?\n\nQuestion: {full_question}\n\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "yahoo_answers_topics"}

            return self._build_pair(
                question=prompt,
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
