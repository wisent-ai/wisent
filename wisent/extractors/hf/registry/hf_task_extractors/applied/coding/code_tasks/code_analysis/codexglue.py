from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "CodexglueExtractor",
    "Code2TextExtractor",
    "Code2TextGoExtractor",
    "Code2TextJavaExtractor",
    "Code2TextJavascriptExtractor",
    "Code2TextPhpExtractor",
    "Code2TextRubyExtractor",
    "DocNliExtractor",
]

log = setup_logger(__name__)


class CodexglueExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for codexglue dataset.

    Schema (code_x_glue_tc_text_to_code):
        - nl: str (natural language description/prompt)
        - code: str (code answer/solution)
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from codexglue examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="code_x_glue_tc_text_to_code",
            dataset_config="default",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} codexglue examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid codexglue pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("nl", "").strip()
            answer = doc.get("code", "")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Convert answer to string
            correct_answer = str(answer).strip()

            # Create incorrect answer (modify or corrupt)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"{question}\n\nGenerate code based on description:"

            metadata = {
                "label": "codexglue",
                "source": "code_x_glue_tc_text_to_code",
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly
        if len(correct) > 10:
            return correct[:len(correct)//2] + "# CORRUPTED" + correct[len(correct)//2:]
        return f"{correct} # INCORRECT"


class Code2TextExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for code2text (code summarization) benchmarks.

    Loads CM/codexglue_code2text_<lang> directly. The lm-eval framework does not
    have a code2text task, so we use the HF dataset.

    Schema:
        - code_tokens: list[str] (tokenized code)
        - docstring_tokens: list[str] (tokenized docstring/summary)
    """

    evaluator_name = "generation"

    # Default to Python; subclasses can override
    LANGUAGE = "python"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name=f"CM/codexglue_code2text_{self.LANGUAGE}",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting code2text pairs from {len(docs)} examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid code2text pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            code_tokens = doc.get("code_tokens", [])
            docstring_tokens = doc.get("docstring_tokens", [])

            if not code_tokens or not docstring_tokens:
                return None

            code = " ".join(str(t) for t in code_tokens).replace("\n", " ")
            code = " ".join(code.strip().split())
            docstring = " ".join(str(t) for t in docstring_tokens).replace("\n", " ")
            docstring = " ".join(docstring.strip().split())

            if not code or not docstring:
                return None

            # Synthetic incorrect: reverse word order
            words = docstring.split()
            incorrect = " ".join(reversed(words)) if len(words) > 1 else "incorrect docstring"

            from wisent.core.primitives.contrastive_pairs.core.io.response import (
                NegativeResponse,
                PositiveResponse,
            )
            return ContrastivePair(
                prompt=f"Generate documentation for this code:\n\n{code}\n\nDocumentation:",
                positive_response=PositiveResponse(model_response=docstring),
                negative_response=NegativeResponse(model_response=incorrect),
                label="code2text",
            )

        except Exception as exc:
            log.error(f"Error extracting code2text pair: {exc}", exc_info=True)
            return None


class Code2TextGoExtractor(Code2TextExtractor):
    LANGUAGE = "go"


class Code2TextJavaExtractor(Code2TextExtractor):
    LANGUAGE = "java"


class Code2TextJavascriptExtractor(Code2TextExtractor):
    LANGUAGE = "javascript"


class Code2TextPhpExtractor(Code2TextExtractor):
    LANGUAGE = "php"


class Code2TextRubyExtractor(Code2TextExtractor):
    LANGUAGE = "ruby"


class DocNliExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for DocNLI (document-level natural language inference)."""

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="saattrupdan/doc-nli",
            split="test",
            limit=max_items,
        )
        from wisent.core.primitives.contrastive_pairs.core.io.response import (
            NegativeResponse,
            PositiveResponse,
        )
        pairs: list[ContrastivePair] = []
        for doc in docs:
            premise = str(doc.get("premise", "")).strip()
            hypothesis = str(doc.get("hypothesis", "")).strip()
            label = str(doc.get("label", "")).strip()
            if not premise or not hypothesis or label not in ("entailment", "not_entailment"):
                continue
            correct = "Yes" if label == "entailment" else "No"
            incorrect = "No" if label == "entailment" else "Yes"
            pairs.append(ContrastivePair(
                prompt=f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis?",
                positive_response=PositiveResponse(model_response=correct),
                negative_response=NegativeResponse(model_response=incorrect),
                label="doc",
            ))
            if max_items is not None and len(pairs) >= max_items:
                break
        return pairs

