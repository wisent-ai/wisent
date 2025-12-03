from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["JsonschemaBenchExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "jsonschema_bench",
    "jsonschema_bench_easy",
    "jsonschema_bench_medium",
    "jsonschema_bench_hard",
)
class JsonschemaBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for JSON Schema Bench benchmark - JSON schema generation task."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from JSON Schema Bench docs.

        Note: This is a JSON schema generation task where models generate
        valid JSON objects conforming to a given schema. Since there are no
        ground truth JSON objects, we use synthetic negatives.

        Note: jsonschema_bench has a misconfigured validation_split ('valid' instead of 'val'),
        so we always use test_docs.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        # Always use test_docs due to misconfigured validation_split in lm-eval task
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc='test')
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid JSON Schema Bench pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        lm_eval_task_data: ConfigurableTask
    ) -> ContrastivePair | None:
        """
        Convert a single JSON Schema Bench doc into a ContrastivePair.

        jsonschema_bench docs only have 'json_schema' and 'unique_id' fields.
        We create synthetic contrastive pairs:
        - Positive: Placeholder for valid conforming JSON
        - Negative: Malformed/invalid JSON
        """
        log = bind(_LOG, doc_id=doc.get("unique_id", "unknown"))

        try:
            # jsonschema_bench only has json_schema and unique_id
            if "json_schema" not in doc:
                log.debug("Skipping doc - missing json_schema field", extra={"doc": doc})
                return None

            json_schema = doc.get("json_schema", "")
            if not json_schema:
                return None

            # Build the prompt using the task's doc_to_text function
            if hasattr(lm_eval_task_data, 'doc_to_text'):
                prompt = lm_eval_task_data.doc_to_text(doc)
            else:
                prompt = f"JSON schema: {json_schema}\n\nJSON object: "

            # Since there's no ground truth JSON object, use synthetic responses:
            # Positive: A minimal valid JSON placeholder
            # Negative: Malformed JSON
            positive = "{}"  # Valid minimal JSON
            negative = "{invalid"  # Malformed JSON

            metadata = {"label": "jsonschema_bench"}

            return self._build_pair(
                question=prompt,
                correct=positive,
                incorrect=negative,
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
