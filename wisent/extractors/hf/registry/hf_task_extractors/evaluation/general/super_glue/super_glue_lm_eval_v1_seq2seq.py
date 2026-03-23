from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.config_tools.constants import INDEX_FIRST, INDEX_SECOND

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SuperGlueLmEvalV1Seq2seqExtractor"]
_LOG = setup_logger(__name__)

task_names = ("super-glue-lm-eval-v1-seq2seq",)
class SuperGlueLmEvalV1Seq2seqExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Super Glue Lm Eval V1 Seq2Seq benchmark."""


    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask | None = None,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="super_glue_lm_eval_v1_seq2seq")
        max_items = self._normalize_limit(limit)

        if lm_eval_task_data is not None:
            docs = self.load_docs(
                lm_eval_task_data, max_items,
                preferred_doc=preferred_doc)
        else:
            docs = self._load_superglue_boolq(max_items)

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs",
                 extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted")

        return pairs

    def _load_superglue_boolq(self, max_items):
        """Load BoolQ from SuperGLUE directly via HuggingFace."""
        return self.load_dataset(
            dataset_name="google/boolq",
            split="validation",
            limit=max_items,
        )

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = (
                doc.get("question", "")
                or doc.get("query", "")
                or doc.get("input", "")
                or doc.get("instruction", "")
                or doc.get("prompt", "")
            ).strip()

            choices = doc.get(
                "choices",
                doc.get("options", doc.get("answers", [])))

            if not choices and "option_a" in doc:
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]

            answer = doc.get(
                "answer",
                doc.get("label", doc.get("target", None)))

            # BoolQ-style: boolean answer, no choices
            if isinstance(answer, bool) and not choices:
                choices = ["no", "yes"]
                answer_idx = INDEX_FIRST if not answer else INDEX_SECOND

            elif isinstance(answer, str) and len(answer) == INDEX_SECOND and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                return None

            if not question or not choices:
                return None
            if not (INDEX_FIRST <= answer_idx < len(choices)):
                return None

            correct = str(choices[answer_idx]).strip()
            inc_idx = (answer_idx + INDEX_SECOND) % len(choices)
            incorrect = str(choices[inc_idx]).strip()

            formatted_q = (
                f"Question: {question}\nA. {incorrect}\nB. {correct}")
            metadata = {"label": "super_glue_lm_eval_v1_seq2seq"}

            return self._build_pair(
                question=formatted_q,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair", exc_info=exc)
            return None

