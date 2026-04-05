from __future__ import annotations

from typing import Any, TYPE_CHECKING
import json

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AgExtractor"]
_LOG = setup_logger(__name__)

task_names = ("ag_news", "ag")

class AgExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Ag benchmark - text classification task."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)
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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: Unitxt format with source/target/task_data
            if "source" in doc and "target" in doc:
                return self._extract_from_unitxt_format(doc, log)

            # Format 2: Standard HuggingFace format with text/label
            if "text" in doc and "label" in doc:
                return self._extract_from_hf_format(doc, log)

            log.debug("Skipping doc - no supported format found", extra={"doc_keys": list(doc.keys())})
            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    def _extract_from_unitxt_format(self, doc: dict[str, Any], log: Any) -> ContrastivePair | None:
        """Extract pair from Unitxt format (source/target/task_data)."""
        source = doc.get("source", "").strip()
        target = doc.get("target", "").strip()

        if not source or not target:
            log.debug("Skipping doc - missing source or target in Unitxt format", extra={"doc": doc})
            return None

        # Try to extract classes from task_data (Unitxt format)
        classes = self._extract_classes_from_task_data(doc)

        # If no classes found, try to extract from source text
        if not classes:
            classes = self._extract_categories_from_source(source)

        if not classes:
            log.debug("Could not extract classes from task_data or source", extra={"source": source})
            return None

        # Verify target is in classes (case-insensitive)
        target_lower = target.lower()
        matched_target = next((cls for cls in classes if cls.lower() == target_lower), None)
        if matched_target is None:
            log.debug("Target not found in classes", extra={"target": target, "classes": classes})
            return None
        target = matched_target

        # Select incorrect answer (any class that's not the target)
        incorrect = next((cls for cls in classes if cls != target), None)
        if not incorrect:
            log.debug("Could not find incorrect class", extra={"target": target, "classes": classes})
            return None

        metadata = {"label": "ag_news"}

        return self._build_pair(
            question=source,
            correct=target,
            incorrect=incorrect,
            metadata=metadata,
        )

    def _extract_from_hf_format(self, doc: dict[str, Any], log: Any) -> ContrastivePair | None:
        """Extract pair from standard HuggingFace format (text/label)."""
        text = doc.get("text", "").strip()
        label = doc.get("label")

        if not text or label is None:
            log.debug("Skipping doc - missing text or label in HF format", extra={"doc": doc})
            return None

        # AG News has 4 categories
        # label: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
        ag_news_categories = ["World", "Sports", "Business", "Sci/Tech"]

        # Convert numeric label to category name
        if isinstance(label, int):
            if 0 <= label < len(ag_news_categories):
                correct = ag_news_categories[label]
            else:
                log.debug("Invalid label value", extra={"label": label})
                return None
        elif isinstance(label, str):
            # If label is already a string, use it directly
            correct = label.strip()
            if correct not in ag_news_categories:
                log.debug("Label not in expected categories", extra={"label": correct, "categories": ag_news_categories})
                return None
        else:
            log.debug("Invalid label type", extra={"label_type": type(label).__name__})
            return None

        # Select an incorrect answer (any category that's not the correct one)
        incorrect = next((cat for cat in ag_news_categories if cat != correct), None)
        if not incorrect:
            log.debug("Could not find incorrect category")
            return None

        metadata = {"label": "ag_news"}

        return self._build_pair(
            question=text,
            correct=correct,
            incorrect=incorrect,
            metadata=metadata,
        )

    @staticmethod
    def _extract_classes_from_task_data(doc: dict[str, Any]) -> list[str]:
        """
        Extract classes from task_data field (Unitxt format).
        """
        try:
            task_data_str = doc.get("task_data", "{}")
            if isinstance(task_data_str, str):
                task_data = json.loads(task_data_str)
            else:
                task_data = task_data_str

            classes = task_data.get("classes", [])
            return classes if isinstance(classes, list) else []
        except Exception:
            return []

    @staticmethod
    def _extract_categories_from_source(source: str) -> list[str]:
        """
        Extract category options from the source prompt.

        AG News format: "Classify the Topic of the following Sentence to one of these options: World, Sports, Business, Sci/Tech."
        """
        # Look for "options:" pattern (case insensitive search but preserve original case)
        if "options:" in source.lower():
            # Find the position and extract from original string (to preserve case)
            lower_source = source.lower()
            idx = lower_source.find("options:")
            if idx != -1:
                # Get text after "options:"
                options_text = source[idx + len("options:"):]
                # Split at the first period or newline to get just the category list
                # Format: " World, Sports, Business, Sci/Tech.\nSentence:..."
                end_idx = len(options_text)
                for delimiter in [".\n", ".\r", ".  ", ". "]:
                    pos = options_text.find(delimiter)
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
                options_text = options_text[:end_idx].strip()

                # Remove trailing period if present
                if options_text.endswith("."):
                    options_text = options_text[:-1]

                # Split by comma and clean up
                categories = [cat.strip() for cat in options_text.split(",") if cat.strip()]
                return categories

        return []

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
