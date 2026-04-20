from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PortugueseBenchExtractor"]
_LOG = setup_logger(__name__)

task_names = ("portuguese_bench",)

class PortugueseBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Portuguese Bench benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Portuguese Bench docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Portuguese Bench.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
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
            log.warning("No valid Portuguese Bench pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Portuguese Bench doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # FLORES translation schema: sentence_<lang>_<script> fields
            sentence_fields = [k for k in doc if k.startswith("sentence_")]
            if len(sentence_fields) >= 2:
                source_text = str(doc.get(sentence_fields[0], "")).strip()
                target_text = str(doc.get(sentence_fields[1], "")).strip()
                if source_text and target_text:
                    import random as _random
                    words = target_text.split()
                    if len(words) > 1:
                        _random.seed(hash(target_text) % (2**32))
                        shuffled = words.copy()
                        _random.shuffle(shuffled)
                        incorrect = " ".join(shuffled)
                        if incorrect == target_text:
                            incorrect = " ".join(words[::-1])
                    else:
                        incorrect = "resposta incorreta"
                    if incorrect == target_text:
                        incorrect = "resposta incorreta"
                    return ContrastivePair(
                        prompt=f"Translate: {source_text}",
                        positive_response=PositiveResponse(model_response=target_text),
                        negative_response=NegativeResponse(model_response=incorrect),
                        label="portuguese_bench_translation",
                    )

            # ASSIN entailment format: premise + hypothesis + entailment_judgment
            if "premise" in doc and "hypothesis" in doc and "entailment_judgment" in doc:
                premise = str(doc.get("premise", "")).strip()
                hypothesis = str(doc.get("hypothesis", "")).strip()
                judgment = doc.get("entailment_judgment", -1)
                if premise and hypothesis and judgment in (0, 1):
                    correct = "Sim" if judgment == 1 else "Não"
                    incorrect = "Não" if judgment == 1 else "Sim"
                    return self._build_pair(
                        question=f"Premissa: {premise}\nHipótese: {hypothesis}\nA premissa implica a hipótese?",
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "portuguese_bench"},
                    )

            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("answer", doc.get("answerKey", ""))
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    answer_idx = int(answer) if answer else 0

            # Format 2: instruction + option_a/b/c/d + answer (MMMLU style)
            elif "instruction" in doc and "option_a" in doc:
                question = str(doc.get("instruction", "")).strip()
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')

            # Format 3: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "portuguese_bench"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "portuguese_bench",
            }

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
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
