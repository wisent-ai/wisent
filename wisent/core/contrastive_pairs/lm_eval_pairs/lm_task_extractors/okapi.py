from __future__ import annotations

from typing import TYPE_CHECKING

from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.super_glue import SuperGlueExtractor

if TYPE_CHECKING:
    pass


__all__ = ["OkapiExtractor"]

task_names = (
    "arc_hr", "arc_sr", "arc_it", "arc_pt", "arc_ml",
    "arc_sk", "arc_ta", "arc_uk", "arc_ru", "arc_de",
    "arc_nl", "arc_mr", "arc_hu", "arc_bn", "arc_zh",
    "arc_hy", "arc_sv", "arc_ar", "arc_es", "arc_id",
    "arc_da", "arc_vi", "arc_te", "arc_eu", "arc_gu",
    "arc_ca", "arc_hi", "arc_ro", "arc_fr", "arc_ne",
    "arc_kn", "hellaswag_it", "hellaswag_sr", "hellaswag_hr", "hellaswag_ml",
    "hellaswag_sk", "hellaswag_pt", "hellaswag_ta", "hellaswag_de", "hellaswag_ru",
    "hellaswag_uk", "hellaswag_nl", "hellaswag_bn", "hellaswag_hu", "hellaswag_mr",
    "hellaswag_hy", "hellaswag_es", "hellaswag_ar", "hellaswag_sv", "hellaswag_da",
    "hellaswag_id", "hellaswag_te", "hellaswag_vi", "hellaswag_eu", "hellaswag_ro",
    "hellaswag_hi", "hellaswag_ca", "hellaswag_gu", "hellaswag_ne", "hellaswag_fr",
    "hellaswag_kn", "m_mmlu_en", "m_mmlu_sk", "m_mmlu_ml", "m_mmlu_pt",
    "m_mmlu_it", "m_mmlu_hr", "m_mmlu_sr", "m_mmlu_de", "m_mmlu_uk",
    "m_mmlu_ru", "m_mmlu_ta", "m_mmlu_nl", "m_mmlu_hy", "m_mmlu_zh",
    "m_mmlu_bn", "m_mmlu_mr", "m_mmlu_hu", "m_mmlu_is", "m_mmlu_es",
    "m_mmlu_ar", "m_mmlu_sv", "m_mmlu_nb", "m_mmlu_te", "m_mmlu_vi",
    "m_mmlu_da", "m_mmlu_id", "m_mmlu_ro", "m_mmlu_ca", "m_mmlu_hi",
    "m_mmlu_gu", "m_mmlu_eu", "m_mmlu_kn", "m_mmlu_ne", "m_mmlu_fr",
    "truthfulqa_vi_mc1", "truthfulqa_id_mc1", "truthfulqa_ml_mc1", "truthfulqa_fr_mc1", "truthfulqa_te_mc2",
    "truthfulqa_it_mc2", "truthfulqa_sr_mc1", "truthfulqa_de_mc1", "truthfulqa_gu_mc1", "truthfulqa_ru_mc1",
    "truthfulqa_hr_mc2", "truthfulqa_bn_mc1", "truthfulqa_sv_mc2", "truthfulqa_ta_mc1", "truthfulqa_hi_mc1",
    "truthfulqa_ro_mc2", "truthfulqa_da_mc2", "truthfulqa_nl_mc2", "truthfulqa_es_mc2", "truthfulqa_mr_mc1",
    "truthfulqa_hy_mc2", "truthfulqa_hu_mc2", "truthfulqa_pt_mc2", "truthfulqa_zh_mc1", "truthfulqa_sk_mc1",
    "truthfulqa_eu_mc2", "truthfulqa_ca_mc2", "truthfulqa_ar_mc1", "truthfulqa_kn_mc2", "truthfulqa_ne_mc1",
    "truthfulqa_uk_mc1", "truthfulqa_uk_mc2", "truthfulqa_ar_mc2", "truthfulqa_ne_mc2", "truthfulqa_kn_mc1",
    "truthfulqa_pt_mc1", "truthfulqa_zh_mc2", "truthfulqa_sk_mc2", "truthfulqa_eu_mc1", "truthfulqa_ca_mc1",
    "truthfulqa_hu_mc1", "truthfulqa_ro_mc1", "truthfulqa_hi_mc2", "truthfulqa_da_mc1", "truthfulqa_nl_mc1",
    "truthfulqa_es_mc1", "truthfulqa_hy_mc1", "truthfulqa_mr_mc2", "truthfulqa_gu_mc2", "truthfulqa_hr_mc1",
    "truthfulqa_ru_mc2", "truthfulqa_bn_mc2", "truthfulqa_sv_mc1", "truthfulqa_ta_mc2", "truthfulqa_de_mc2",
    "truthfulqa_vi_mc2", "truthfulqa_id_mc2", "truthfulqa_ml_mc2", "truthfulqa_fr_mc2", "truthfulqa_te_mc1",
    "truthfulqa_sr_mc2", "truthfulqa_it_mc1",
)
evaluator_name = "log_likelihoods"


class OkapiExtractor(SuperGlueExtractor):
    """Extractor for Okapi multilingual benchmarks (arc, hellaswag, mmlu, truthfulqa).

    Okapi benchmarks use the same multiple-choice format as SuperGlue, so we inherit
    the extraction logic directly from SuperGlueExtractor.
    """
    pass
