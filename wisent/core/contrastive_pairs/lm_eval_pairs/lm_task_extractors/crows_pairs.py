from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CrowsPairsExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "crows_pairs",
    "crows_pairs_english",
    "crows_pairs_english_religion",
    "crows_pairs_english_nationality",
    "crows_pairs_english_race_color",
    "crows_pairs_english_age",
    "crows_pairs_english_gender",
    "crows_pairs_english_disability",
    "crows_pairs_english_sexual_orientation",
    "crows_pairs_english_physical_appearance",
    "crows_pairs_english_socioeconomic",
    "crows_pairs_english_autre",
    "crows_pairs_french",
    "crows_pairs_french_religion",
    "crows_pairs_french_nationality",
    "crows_pairs_french_race_color",
    "crows_pairs_french_age",
    "crows_pairs_french_gender",
    "crows_pairs_french_disability",
    "crows_pairs_french_sexual_orientation",
    "crows_pairs_french_physical_appearance",
    "crows_pairs_french_socioeconomic",
    "crows_pairs_french_autre",
)
class CrowsPairsExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Crows Pairs benchmark - bias measurement task."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Crows Pairs docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Crows Pairs.
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
            log.warning("No valid Crows Pairs pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Crows Pairs doc into a ContrastivePair.

        Crows Pairs format:
        - sent_more: One sentence (could be stereotypical or anti-stereotypical)
        - sent_less: Another sentence (the pair)
        - stereo_antistereo: "stereo" if sent_more is stereotypical, "antistereo" if sent_more is anti-stereotypical
        - bias_type: Type of bias

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            sent_more = doc.get("sent_more", "").strip()
            sent_less = doc.get("sent_less", "").strip()
            stereo_antistereo = doc.get("stereo_antistereo", "").strip()

            if not sent_more or not sent_less or not stereo_antistereo:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Determine which sentence is stereotypical and which is anti-stereotypical
            # For bias measurement, we want the model to prefer anti-stereotypical sentences
            if stereo_antistereo == "stereo":
                # sent_more is stereotypical (negative), sent_less is anti-stereotypical (positive)
                positive_sentence = sent_less
                negative_sentence = sent_more
            elif stereo_antistereo == "antistereo":
                # sent_more is anti-stereotypical (positive), sent_less is stereotypical (negative)
                positive_sentence = sent_more
                negative_sentence = sent_less
            else:
                log.debug(
                    "Unknown stereo_antistereo value",
                    extra={"stereo_antistereo": stereo_antistereo, "doc": doc},
                )
                return None

            # For log-likelihood evaluation, the prompt can be simple or empty
            # The evaluator will compare the probabilities of the two sentences
            prompt = "Choose the more appropriate sentence"

            metadata = {
                "label": "crows_pairs",
                "bias_type": doc.get("bias_type", "unknown"),
            }

            return self._build_pair(
                question=prompt,
                correct=positive_sentence,
                incorrect=negative_sentence,
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
