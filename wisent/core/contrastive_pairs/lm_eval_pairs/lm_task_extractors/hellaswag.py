from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["HellaSwagExtractor"]
_LOG = setup_logger(__name__)

task_names = ("hellaswag",)

class HellaSwagExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the HellaSwag benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HellaSwag docs.

        HellaSwag schema:
            - query: str
            - endings: list of str
            - label: index of correct ending, str
        
        Args:
            lm_eval_task_data: lm-eval task instance for HellaSwag.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

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
            log.warning("No valid HellaSwag pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Hellaswag doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        IMPROVED: Uses the HARDEST incorrect ending (most similar to correct)
        to create maximum contrast signal for directional modification.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            query = str(doc.get("query", "")).strip()
            endings = doc.get("endings", [])
            label = str(doc.get("label", "")).strip()
            label = int(label)

            if not query or not endings or not (0 <= label < len(endings)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = endings[label]

            # Get ALL incorrect endings
            incorrect_endings = [e for i, e in enumerate(endings) if i != label]

            # Use the HARDEST incorrect (longest, as proxy for most plausible)
            # This creates maximum contrast for the steering vector
            incorrect = max(incorrect_endings, key=len) if incorrect_endings else endings[(label+1)%len(endings)]

            question = f"{query}"
            prompt = f"{question}"

            metadata = {
                "label": "hellaswag",
            }

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