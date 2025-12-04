from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MedicalAbstractsExtractor"]
_LOG = setup_logger(__name__)

task_names = ("medical_abstracts",)

class MedicalAbstractsExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Medical Abstracts benchmark - medical topic classification."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Unitxt format
            if "source" in doc and "target" in doc and "task_data" in doc:
                import json

                source = str(doc.get("source", "")).strip()
                target = str(doc.get("target", "")).strip()
                task_data_str = doc.get("task_data", "{}")

                if not source or not target:
                    log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                    return None

                # Parse task_data to get classes
                task_data = json.loads(task_data_str)
                classes = task_data.get("classes", [])

                if not classes:
                    log.debug("Skipping doc due to missing classes", extra={"doc": doc})
                    return None

                prompt = source
                correct = target

                # Find an incorrect class
                incorrect = None
                for cls in classes:
                    if cls != target:
                        incorrect = cls
                        break

                if not incorrect:
                    log.debug("No incorrect class found", extra={"doc": doc})
                    return None

                metadata = {"label": "medical_abstracts"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

