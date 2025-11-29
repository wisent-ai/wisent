from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EthosExtractor"]
_LOG = setup_logger(__name__)

task_names = ("ethos", "ethos_binary")

class EthosExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Ethos benchmark - hate speech classification task."""


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

        # Ethos format: source (prompt) + target (classification label)
        # Group by target label
        hate_speech_texts = []
        not_hate_speech_texts = []

        for doc in docs:
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip().lower()

            if not source or not target:
                continue

            # Extract the actual text from the source (it contains the full prompt)
            # Format: "Classify... Sentence:\n<text>\nHate Speech:\n"
            if "Sentence:" in source:
                text_part = source.split("Sentence:")[1].strip()
                # Remove the "Hate Speech:" part if present
                if "\nHate Speech:" in text_part:
                    text_part = text_part.split("\nHate Speech:")[0].strip()
                elif "\n" in text_part:
                    text_part = text_part.split("\n")[0].strip()

                if "hate speech" in target and "not" not in target:
                    hate_speech_texts.append((source, target))
                elif "not hate speech" in target:
                    not_hate_speech_texts.append((source, target))

        # Create pairs
        num_pairs = min(len(hate_speech_texts), len(not_hate_speech_texts))
        if max_items is not None:
            num_pairs = min(num_pairs, max_items)

        for i in range(num_pairs):
            # Use source (full prompt) and target (correct label)
            source_not_hate, target_not_hate = not_hate_speech_texts[i]
            source_hate, target_hate = hate_speech_texts[i]

            # Use the same prompt format for both
            metadata = {"label": "ethos"}

            pair = self._build_pair(
                question=source_not_hate,
                correct=target_not_hate,
                incorrect=target_hate,
                metadata=metadata,
            )
            pairs.append(pair)

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

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
