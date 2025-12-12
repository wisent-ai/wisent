from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Code2TextExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "code2text",
    "code2text_go",
    "code2text_java",
    "code2text_javascript",
    "code2text_php",
    "code2text_python",
    "code2text_ruby"
)

class Code2TextExtractor(LMEvalBenchmarkExtractor):
    """Extractor for code2text tasks - generates documentation from code."""


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
            # Format: Code2Text format (code_tokens + docstring_tokens)
            if "code_tokens" in doc and "docstring_tokens" in doc:
                code_tokens = doc.get("code_tokens", [])
                docstring_tokens = doc.get("docstring_tokens", [])

                if not code_tokens or not docstring_tokens:
                    log.debug("Skipping doc - empty code or docstring tokens", extra={"doc": doc})
                    return None

                # Join tokens to create code and docstring
                code = " ".join(code_tokens).replace("\n", " ")
                code = " ".join(code.strip().split())

                docstring = " ".join(docstring_tokens).replace("\n", " ")
                docstring = " ".join(docstring.strip().split())

                # Create synthetic negative by shuffling words in the docstring
                import random
                words = docstring.split()
                if len(words) > 3:
                    shuffled_words = words.copy()
                    random.shuffle(shuffled_words)

                    # If shuffle resulted in same order, reverse it
                    if shuffled_words == words:
                        shuffled_words = list(reversed(words))

                    incorrect_docstring = " ".join(shuffled_words)
                else:
                    # For very short docstrings, just use a generic incorrect one
                    incorrect_docstring = "This is an incorrect docstring."

                prompt = f"Generate documentation for this code:\n\n{code}\n\nDocumentation:"

                metadata = {"label": "code2text"}

                return self._build_pair(
                    question=prompt,
                    correct=docstring,
                    incorrect=incorrect_docstring,
                    metadata=metadata,
                )

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
