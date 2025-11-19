"""Truthfulqa multi group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

TRUTHFULQA_MULTI_TASKS = {
    "truthfulqa-multi": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_gen_ca": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_gen_en": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_gen_es": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_gen_eu": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_gen_gl": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc1_ca": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc1_en": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc1_es": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc1_eu": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc1_gl": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc2_ca": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc2_en": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc2_es": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc2_eu": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
    "truthfulqa-multi_mc2_gl": f"{BASE_IMPORT}truthfulqa_multi:TruthfulqaMultiExtractor",
}
