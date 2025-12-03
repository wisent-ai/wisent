from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TranslationExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "translation",
    "gpt3_translation_benchmarks",
    "iwslt2017-ar-en",
    "iwslt2017-en-ar",
    "wmt14-en-fr",
    "wmt14-fr-en",
    "wmt16-de-en",
    "wmt16-en-de",
    "wmt16-en-ro",
    "wmt16-ro-en",
)

class TranslationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Translation benchmark (WMT, IWSLT).

    Translation format: {'translation': {'en': 'English text', 'fr': 'French text'}}

    For contrastive pairs:
    - Positive: Correct translation from reference
    - Negative: Word-shuffled version to simulate bad translation
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Translation docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Translation.
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
            log.warning("No valid Translation pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Translation doc into a ContrastivePair.

        Translation format: {'translation': {'en': 'English text', 'fr': 'French text'}}

        We extract source->target direction based on the task name pattern.
        For example, wmt14-en-fr translates from 'en' to 'fr'.
        """
        if "translation" not in doc:
            return None

        translation_dict = doc["translation"]
        if not isinstance(translation_dict, dict) or len(translation_dict) < 2:
            return None

        # Determine source and target languages from the keys
        lang_keys = list(translation_dict.keys())
        if len(lang_keys) != 2:
            return None

        # Try to infer direction from task name if available in context
        # Default: first key is source, second is target
        source_lang = lang_keys[0]
        target_lang = lang_keys[1]

        source_text = str(translation_dict[source_lang]).strip()
        target_text = str(translation_dict[target_lang]).strip()

        if not source_text or not target_text:
            return None

        # Create synthetic negative by shuffling words in target
        import random
        target_words = target_text.split()
        if len(target_words) > 1:
            shuffled_words = target_words.copy()
            # Shuffle until different
            max_attempts = 10
            for _ in range(max_attempts):
                random.shuffle(shuffled_words)
                if shuffled_words != target_words:
                    break
            incorrect_translation = " ".join(shuffled_words)
        else:
            # Single word: just add "wrong " prefix
            incorrect_translation = f"wrong {target_text}"

        prompt = f"Translate from {source_lang} to {target_lang}:\n{source_text}"

        metadata = {"label": "translation", "source_lang": source_lang, "target_lang": target_lang}

        return self._build_pair(
            question=prompt,
            correct=target_text,
            incorrect=incorrect_translation,
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
