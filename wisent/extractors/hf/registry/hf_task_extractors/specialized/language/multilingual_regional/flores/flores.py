from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FloresExtractor"]
_LOG = setup_logger(__name__)

from wisent.extractors.hf.hf_task_extractors.flores_names_a_to_e import FLORES_NAMES_A_TO_E
from wisent.extractors.hf.hf_task_extractors.flores_names_f_to_z import FLORES_NAMES_F_TO_Z

task_names = FLORES_NAMES_A_TO_E + FLORES_NAMES_F_TO_Z

class FloresExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Flores benchmark - multilingual machine translation tasks."""


    evaluator_name = "generation"
    
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="flores")
        max_items = self._normalize_limit(limit)
        
        # Load data directly from HuggingFace
        from datasets import load_dataset
        try:
            # Try to load from cache (trust_remote_code no longer supported)
            ds = load_dataset("facebook/flores", "all", split="devtest")
            docs = list(ds)
            if max_items:
                docs = docs[:max_items]
        except Exception as e:
            log.error(f"Failed to load flores dataset: {e}")
            return []
        
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "flores"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Flores doc into a ContrastivePair.

        Flores format:
        - sentence_{source_lang}_{script}: source text
        - sentence_{target_lang}_{script}: target text

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Find sentence fields (format: sentence_{lang}_{script})
            sentence_fields = [k for k in doc.keys() if k.startswith("sentence_")]

            if len(sentence_fields) < 2:
                log.debug("Skipping doc due to missing sentence fields", extra={"doc": doc})
                return None

            # Get source and target sentences
            # Usually first is source, second is target
            source_field = sentence_fields[0]
            target_field = sentence_fields[1]

            source_text = doc.get(source_field, "").strip()
            target_text = doc.get(target_field, "").strip()

            if not source_text or not target_text:
                log.debug("Skipping doc due to empty text", extra={"doc": doc})
                return None

            # Extract language codes for prompt
            # Format: sentence_afr_Latn → afr_Latn
            source_lang = source_field.replace("sentence_", "")
            target_lang = target_field.replace("sentence_", "")

            # Create translation prompt
            prompt = f"Translate the following from {source_lang} to {target_lang}:\n{source_text}"

            # Positive: correct translation
            correct_translation = target_text

            # Negative: shuffled words for synthetic incorrect translation
            words = target_text.split()
            if len(words) < 2:
                # For single-word translations, use a placeholder
                incorrect_translation = "[incorrect translation]"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_translation = ' '.join(shuffled_words)

            metadata = {"label": "flores"}

            return self._build_pair(
                question=prompt,
                correct=correct_translation,
                incorrect=incorrect_translation,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

