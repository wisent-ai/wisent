from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["CodexglueCodeToTextRubyExtractor"]

log = logging.getLogger(__name__)

task_names = ("codexglue_code_to_text_ruby",)

class CodexglueCodeToTextRubyExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for CodeXGLUE code-to-text Ruby."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("code_x_glue_ct_code_to_text", "ruby", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load codexglue_code_to_text_ruby dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting {len(dataset)} codexglue_code_to_text_ruby pairs")

        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)
                if max_items and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            code = doc.get("code", "").strip()
            docstring = doc.get("docstring", "").strip()

            if not code or not docstring:
                return None

            correct = docstring
            incorrect = "This is an incorrect description."

            question = f"Describe what this Ruby code does:\n\n{code}"
            metadata = {"label": "codexglue_code_to_text_ruby"}

            return self._build_pair(question, correct, incorrect, metadata)
        except Exception as e:
            log.error(f"Error extracting pair: {e}")
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str, metadata: dict[str, Any] | None = None) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
