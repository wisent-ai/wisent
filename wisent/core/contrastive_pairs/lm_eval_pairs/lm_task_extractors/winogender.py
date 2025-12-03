from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["WinogenderExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "winogender",
    "winogender_all", "winogender_female", "winogender_gotcha", "winogender_gotcha_female",
    "winogender_gotcha_male", "winogender_male", "winogender_neutral",
)

class WinogenderExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Winogender benchmark (pronoun resolution with gender bias).

    Format: {
        sentence: str,  # Sentence with pronoun
        pronoun: str,   # The pronoun to resolve
        occupation: str,  # The occupation entity
        participant: str, # The participant entity
        target: str,    # Correct referent (occupation or participant)
        label: int      # 1 if target is participant, 0 if occupation
    }

    For contrastive pairs:
    - Positive: Correct referent (target)
    - Negative: Incorrect referent (the other entity)
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Winogender docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Winogender.
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
            log.warning("No valid Winogender pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Winogender doc into a ContrastivePair.

        Winogender format: {
            sentence: str,
            pronoun: str,
            occupation: str,
            participant: str,
            target: str,  # correct referent
            label: int    # 1 if participant, 0 if occupation
        }
        """
        if "sentence" not in doc or "target" not in doc:
            return None

        sentence = str(doc["sentence"]).strip()
        pronoun = str(doc.get("pronoun", "")).strip()
        occupation = str(doc.get("occupation", "")).strip()
        participant = str(doc.get("participant", "")).strip()
        target = str(doc["target"]).strip()

        if not sentence or not target or not occupation or not participant:
            return None

        # The correct answer is the target
        correct = target

        # The incorrect answer is the other entity
        if target == occupation:
            incorrect = participant
        else:
            incorrect = occupation

        # Create prompt asking what the pronoun refers to
        prompt = f'In the sentence "{sentence}", what does "{pronoun}" refer to?'

        metadata = {
            "label": "winogender",
            "gender": doc.get("gender", "unknown"),
        }

        return self._build_pair(
            question=prompt,
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
