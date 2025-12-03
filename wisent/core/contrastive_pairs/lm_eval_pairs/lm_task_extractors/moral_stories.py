from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MoralStoriesExtractor"]
_LOG = setup_logger(__name__)

task_names = ("moral_stories",)

class MoralStoriesExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Moral Stories benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Moral Stories docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Moral Stories.
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
            log.warning("No valid Moral Stories pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Moral Stories doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Moral Stories format (after process_docs):
        - query: "{norm} {situation} {intention}" (context)
        - choices: [moral_action, immoral_action]
        - label: 0 (always the first choice is correct)

        Raw format:
        - norm, situation, intention: text fields
        - moral_action: correct choice
        - immoral_action: incorrect choice
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: Processed format (query + choices + label)
            if "query" in doc and "choices" in doc and "label" in doc:
                query = str(doc.get("query", "")).strip()
                choices = doc.get("choices", [])
                label = doc.get("label", 0)

                if not query or not choices or len(choices) < 2:
                    log.debug("Skipping doc with missing query/choices", extra={"doc": doc})
                    return None

                if not isinstance(label, int) or not (0 <= label < len(choices)):
                    log.debug("Invalid label", extra={"label": label, "doc": doc})
                    return None

                # Format prompt exactly as lm-eval does (just the query)
                prompt = query

                correct = str(choices[label]).strip()
                incorrect_idx = (label + 1) % len(choices)
                incorrect = str(choices[incorrect_idx]).strip()

                metadata = {"label": "moral_stories"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 2: Raw format (norm + situation + intention + moral_action + immoral_action)
            elif "norm" in doc and "situation" in doc and "intention" in doc and "moral_action" in doc and "immoral_action" in doc:
                norm = str(doc.get("norm", "")).strip().capitalize()
                situation = str(doc.get("situation", "")).strip().capitalize()
                intention = str(doc.get("intention", "")).strip().capitalize()
                moral_action = str(doc.get("moral_action", "")).strip()
                immoral_action = str(doc.get("immoral_action", "")).strip()

                if not norm or not situation or not intention or not moral_action or not immoral_action:
                    log.debug("Skipping doc with missing fields", extra={"doc": doc})
                    return None

                # Construct query same way as process_docs
                query = f"{norm} {situation} {intention}"

                metadata = {"label": "moral_stories"}

                return self._build_pair(
                    question=query,
                    correct=moral_action,
                    incorrect=immoral_action,
                    metadata=metadata,
                )

            log.debug("Skipping doc without required fields", extra={"doc": doc})
            return None

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
