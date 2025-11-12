from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FldExtractor"]
_LOG = setup_logger(__name__)


class FldExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Fld benchmark - Formal Logic Deduction tasks.

    FLD evaluates logical reasoning by presenting facts and hypotheses,
    requiring models to determine validity (proved/disproved/unknown).
    Includes default and logical_formula variants with standard and star difficulty.
    """

    task_names = (
        "fld_default",
        "fld_logical_formula_default",
        "fld_logical_formula_star",
        "fld_star",
    )
    evaluator_name = "exact_match"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Fld docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Fld.
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
            log.warning("No valid Fld pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Fld doc into a ContrastivePair, if possible.

        FLD format:
        - hypothesis: the statement to be proved/disproved
        - context: the facts/premises
        - proof_label: one of PROVED, DISPROVED, UNKNOWN

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # FLD-specific fields
            hypothesis = doc.get("hypothesis", "").strip()
            context = doc.get("context", "").strip()
            proof_label = doc.get("proof_label", "").strip().upper()

            if not hypothesis or not context or not proof_label:
                log.debug(
                    "Skipping doc due to missing required FLD fields",
                    extra={"doc": doc},
                )
                return None

            # Valid labels for FLD
            valid_labels = ["PROVED", "DISPROVED", "UNKNOWN"]
            if proof_label not in valid_labels:
                log.debug(
                    "Skipping doc due to invalid proof_label",
                    extra={"proof_label": proof_label, "doc": doc},
                )
                return None

            # Create prompt from hypothesis and context
            prompt = f"Given the following facts:\n{context}\n\nDetermine if the hypothesis can be proved, disproved, or is unknown:\nHypothesis: {hypothesis}\n\nAnswer (PROVED/DISPROVED/UNKNOWN):"

            # Positive response: correct label
            correct_answer = proof_label

            # Negative response: pick a different label
            incorrect_options = [label for label in valid_labels if label != proof_label]
            incorrect_answer = incorrect_options[0] if incorrect_options else "PROVED"

            metadata = {
                "label": "fld",
                "proof_label": proof_label,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
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
