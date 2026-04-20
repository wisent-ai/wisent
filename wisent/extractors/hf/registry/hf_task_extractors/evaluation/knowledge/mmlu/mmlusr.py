"""Extractor for MMLU-SR (Stress-Testing) benchmarks from HuggingFace."""

from __future__ import annotations

from typing import Any

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

__all__ = ["MMLUSRExtractor"]
_LOG = setup_logger(__name__)


class MMLUSRExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for MMLU-SR (Stress-Testing) benchmarks."""

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from MMLU-SR dataset.

        Args:
            limit: Maximum number of pairs to extract.

        Returns:
            List of contrastive pairs.
        """
        # Use task_name to route to the correct dataset_config.
        # mmlusr_answer_only_abstract_algebra -> answer_only_abstract_algebra
        task_name = getattr(self, "task_name", "")
        if task_name.startswith("mmlusr_"):
            cfg = task_name[len("mmlusr_"):]
        else:
            cfg = "answer_only_abstract_algebra"
        from datasets import load_dataset
        ds = load_dataset(
            "NiniCat/MMLU-SR",
            cfg,
            split="test",
            revision="bd6827d2542e9d6cf1d58ce262ce0b9ad7742754",
            trust_remote_code=True,
        )
        docs = list(ds)[:limit] if limit else list(ds)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single MMLU-SR doc into a ContrastivePair."""
        log = bind(_LOG, doc_id=doc.get("__index__", "unknown"))

        try:
            # Real MMLU-SR doc fields: question, choice1..choice4, answer (int 0-3)
            question = str(doc.get("question", doc.get("column_0", ""))).strip()
            choices = [
                str(doc.get("choice1", doc.get("column_1", ""))).strip(),
                str(doc.get("choice2", doc.get("column_2", ""))).strip(),
                str(doc.get("choice3", doc.get("column_3", ""))).strip(),
                str(doc.get("choice4", doc.get("column_4", ""))).strip(),
            ]
            raw_answer = doc.get("answer", doc.get("column_5", ""))
            if isinstance(raw_answer, int):
                answer_letter = chr(ord('A') + raw_answer)
            else:
                s = str(raw_answer).strip()
                if s.isdigit():
                    answer_letter = chr(ord('A') + int(s))
                else:
                    answer_letter = s.upper()

            if not question or not all(choices) or not answer_letter:
                log.debug("Skipping: missing question, choices, or answer")
                return None

            # Convert answer letter to index
            if answer_letter not in ['A', 'B', 'C', 'D']:
                log.debug(f"Skipping: invalid answer letter '{answer_letter}'")
                return None

            answer_index = ord(answer_letter) - ord('A')

            if not (0 <= answer_index < len(choices)):
                log.debug(f"Skipping: answer index {answer_index} out of range")
                return None

            correct_answer = choices[answer_index]

            # Get an incorrect answer (any other option)
            incorrect_index = (answer_index + 1) % len(choices)
            incorrect_answer = choices[incorrect_index]

            # Format question with options
            formatted_question = f"Question: {question}\nOptions:\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(ord('A') + i)}. {choice}\n"
            formatted_question += "Answer:"

            metadata = {
                "label": "mmlusr",
            }

            positive_response = PositiveResponse(model_response=correct_answer)
            negative_response = NegativeResponse(model_response=incorrect_answer)

            return ContrastivePair(
                prompt=formatted_question.strip(),
                positive_response=positive_response,
                negative_response=negative_response,
                label=metadata.get("label"),
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None
