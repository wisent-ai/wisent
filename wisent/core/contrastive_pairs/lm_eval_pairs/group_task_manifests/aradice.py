"""AraDICE group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

ARADICE_TASKS = {
    "AraDiCE_ArabicMMLU_lev": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_ArabicMMLU_egy": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_boolq_egy": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_boolq_eng": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_boolq_lev": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_boolq_msa": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_egypt_cultural": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_jordan_cultural": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_lebanon_cultural": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_palestine_cultural": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_qatar_cultural": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_syria_cultural": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_openbookqa_egy": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_openbookqa_eng": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_openbookqa_lev": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_openbookqa_msa": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_piqa_egy": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_piqa_eng": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_piqa_lev": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_piqa_msa": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_truthfulqa_mc1_egy": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_truthfulqa_mc1_eng": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_truthfulqa_mc1_lev": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_truthfulqa_mc1_msa": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_winogrande_egy": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_winogrande_eng": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_winogrande_lev": f"{BASE_IMPORT}aradice:AradiceExtractor",
    "AraDiCE_winogrande_msa": f"{BASE_IMPORT}aradice:AradiceExtractor",
}
