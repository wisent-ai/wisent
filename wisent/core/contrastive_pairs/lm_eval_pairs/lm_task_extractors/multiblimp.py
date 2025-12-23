from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MultiblimpExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "multiblimp_abk", "multiblimp_aln", "multiblimp_amh", "multiblimp_apu", "multiblimp_aqz", "multiblimp_arb",
    "multiblimp_azz", "multiblimp_bel", "multiblimp_ben", "multiblimp_bho", "multiblimp_bor", "multiblimp_bre",
    "multiblimp_bua", "multiblimp_bul", "multiblimp_cat", "multiblimp_ces", "multiblimp_chu", "multiblimp_cym",
    "multiblimp_dan", "multiblimp_deu", "multiblimp_egy", "multiblimp_ell", "multiblimp_eng", "multiblimp_est",
    "multiblimp_eus", "multiblimp_fao", "multiblimp_fas", "multiblimp_fin", "multiblimp_fra", "multiblimp_frm",
    "multiblimp_fro", "multiblimp_gla", "multiblimp_gle", "multiblimp_glg", "multiblimp_got", "multiblimp_grc",
    "multiblimp_guj", "multiblimp_hbo", "multiblimp_hbs", "multiblimp_heb", "multiblimp_hin", "multiblimp_hit",
    "multiblimp_hsb", "multiblimp_hun", "multiblimp_hye", "multiblimp_hyw", "multiblimp_isl", "multiblimp_ita",
    "multiblimp_kat", "multiblimp_kaz", "multiblimp_kir", "multiblimp_kmr", "multiblimp_koi", "multiblimp_kpv",
    "multiblimp_krl", "multiblimp_kxh", "multiblimp_lat", "multiblimp_lav", "multiblimp_lij", "multiblimp_lit",
    "multiblimp_mar", "multiblimp_mdf", "multiblimp_mkd", "multiblimp_myv", "multiblimp_nds", "multiblimp_nhi",
    "multiblimp_nld", "multiblimp_olo", "multiblimp_orv", "multiblimp_ota", "multiblimp_pcm", "multiblimp_pol",
    "multiblimp_por", "multiblimp_quc", "multiblimp_ron", "multiblimp_rus", "multiblimp_sah", "multiblimp_san",
    "multiblimp_slk", "multiblimp_slv", "multiblimp_sme", "multiblimp_sms", "multiblimp_spa", "multiblimp_sqi",
    "multiblimp_swe", "multiblimp_tam", "multiblimp_tpn", "multiblimp_ttc", "multiblimp_tur", "multiblimp_uig",
    "multiblimp_ukr", "multiblimp_urb", "multiblimp_urd", "multiblimp_uzb", "multiblimp_vep", "multiblimp_wbp",
    "multiblimp_wol", "multiblimp_xcl", "multiblimp_xnr", "multiblimp_xpg", "multiblimp_yrl"
)

class MultiblimpExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Multiblimp benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Multiblimp docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Multiblimp.
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
            log.warning("No valid Multiblimp pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Multiblimp doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Multiblimp format (linguistic minimal pairs):
        - sen: grammatically correct sentence
        - wrong_sen: grammatically incorrect sentence
        - Target is always 0 (correct sentence)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Multiblimp format: sen (correct) and wrong_sen (incorrect)
            if "sen" in doc and "wrong_sen" in doc:
                correct_sentence = str(doc.get("sen", "")).strip()
                incorrect_sentence = str(doc.get("wrong_sen", "")).strip()

                if not correct_sentence or not incorrect_sentence:
                    log.debug("Skipping doc with missing sen/wrong_sen", extra={"doc": doc})
                    return None

                # Raw prompt without A./B. formatting
                prompt = "Which sentence is grammatically correct?"

                metadata = {"label": "multiblimp"}

                return self._build_pair(
                    question=prompt,
                    correct=correct_sentence,
                    incorrect=incorrect_sentence,
                    metadata=metadata,
                )

            log.debug("Skipping doc without sen/wrong_sen fields", extra={"doc": doc})
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
