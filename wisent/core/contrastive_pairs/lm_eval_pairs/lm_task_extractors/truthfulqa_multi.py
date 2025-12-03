from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TruthfulqaMultiExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "truthfulqa_multi",
    "truthfulqa_multilingual",
    "truthfulqa-multi",
    "truthfulqa-multi_gen_ca", "truthfulqa-multi_gen_en", "truthfulqa-multi_gen_es",
    "truthfulqa-multi_gen_eu", "truthfulqa-multi_gen_gl", "truthfulqa-multi_mc1_ca",
    "truthfulqa-multi_mc1_en", "truthfulqa-multi_mc1_es", "truthfulqa-multi_mc1_eu",
    "truthfulqa-multi_mc1_gl", "truthfulqa-multi_mc2_ca", "truthfulqa-multi_mc2_en",
    "truthfulqa-multi_mc2_es", "truthfulqa-multi_mc2_eu", "truthfulqa-multi_mc2_gl",
)

# Mixed evaluator - has both gen (generation) and mc (log_likelihoods) variants
# For _gen_ tasks use generation, for _mc_ tasks use log_likelihoods
class TruthfulqaMultiExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the TruthfulQA Multi benchmark (multilingual variant).

    Format: {
        question: str,
        correct_answers: list[str],
        incorrect_answers: list[str],
        mc1_targets: {choices: list[str], labels: list[int]},
        mc2_targets: {choices: list[str], labels: list[int]},
        lang: str
    }

    For contrastive pairs:
    - Positive: Random correct answer from correct_answers
    - Negative: Random incorrect answer from incorrect_answers
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Truthfulqa Multi docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Truthfulqa Multi.
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
            log.warning("No valid Truthfulqa Multi pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA Multi doc into a ContrastivePair.

        TruthfulQA format: {
            question: str,
            correct_answers: list[str],
            incorrect_answers: list[str],
            ...
        }
        """
        if "question" not in doc:
            return None

        question = str(doc["question"]).strip()

        # Get correct and incorrect answers
        correct_answers = doc.get("correct_answers", [])
        incorrect_answers = doc.get("incorrect_answers", [])

        if not correct_answers or not incorrect_answers:
            return None

        # Use the first correct and first incorrect answer
        import random
        correct = random.choice(correct_answers).strip()
        incorrect = random.choice(incorrect_answers).strip()

        if not correct or not incorrect:
            return None

        formatted_question = f"Question: {question}"

        metadata = {
            "label": "truthfulqa-multi",
            "lang": doc.get("lang", "en"),
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
