from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MmmuExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "mmmu", "mmmu_val",
    "mmmu_val_accounting", "mmmu_val_agriculture", "mmmu_val_architecture_and_engineering",
    "mmmu_val_art", "mmmu_val_art_and_design", "mmmu_val_art_theory",
    "mmmu_val_basic_medical_science", "mmmu_val_biology", "mmmu_val_business",
    "mmmu_val_chemistry", "mmmu_val_clinical_medicine", "mmmu_val_computer_science",
    "mmmu_val_design", "mmmu_val_diagnostics_and_laboratory_medicine",
    "mmmu_val_economics", "mmmu_val_electronics", "mmmu_val_energy_and_power",
    "mmmu_val_finance", "mmmu_val_geography", "mmmu_val_health_and_medicine",
    "mmmu_val_history", "mmmu_val_humanities_and_social_science", "mmmu_val_literature",
    "mmmu_val_manage", "mmmu_val_marketing", "mmmu_val_materials", "mmmu_val_math",
    "mmmu_val_mechanical_engineering", "mmmu_val_music", "mmmu_val_pharmacy",
    "mmmu_val_physics", "mmmu_val_psychology", "mmmu_val_public_health",
    "mmmu_val_science", "mmmu_val_sociology", "mmmu_val_tech_and_engineering",
)

class MmmuExtractor(LMEvalBenchmarkExtractor):
    evaluator_name = "generation"
    """Extractor for the MMMU (Multimodal) benchmark.

    Format: {
        question: str,  # May contain "<image N>" placeholders
        options: list[str],  # Multiple choice options
        answer: str,  # Letter answer (A, B, C, D)
        image_1, image_2, etc: PIL Image objects (ignored in text-only extraction)
    }

    NOTE: MMMU is a multimodal benchmark that includes images.
    This extractor only processes the text portion and ignores images.
    For contrastive pairs:
    - Positive: Correct option based on answer
    - Negative: Random incorrect option
    """

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mmmu docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mmmu.
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
            log.warning("No valid Mmmu pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MMMU doc into a ContrastivePair.

        MMMU format: {
            question: str,
            options: list[str],
            answer: str  # Letter like "A", "B", "C", "D"
        }
        """
        if "question" not in doc or "options" not in doc or "answer" not in doc:
            return None

        question = str(doc["question"]).strip()
        options = doc["options"]
        answer = str(doc["answer"]).strip().upper()

        if not question or not options or not answer:
            return None

        # Convert letter answer to index
        if len(answer) == 1 and answer.isalpha():
            answer_idx = ord(answer) - ord('A')
        else:
            return None

        if answer_idx < 0 or answer_idx >= len(options):
            return None

        # Get correct and random incorrect option
        correct = str(options[answer_idx]).strip()

        # Get an incorrect option (next one, wrapping around)
        incorrect_idx = (answer_idx + 1) % len(options)
        incorrect = str(options[incorrect_idx]).strip()

        # Format question (note: images are referenced as "<image N>" but we ignore them)
        formatted_question = f"Question: {question}"

        metadata = {
            "label": "mmmu",
            "subfield": doc.get("subfield", "unknown"),
        }

        return self._build_pair(
            question=formatted_question,
            correct=correct,
            incorrect=incorrect,
            metadata=metadata,
        )

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
