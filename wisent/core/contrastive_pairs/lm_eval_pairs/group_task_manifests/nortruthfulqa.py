"""Nortruthfulqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

NORTRUTHFULQA_TASKS = {
    "nortruthfulqa_gen_nno": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nno_p0": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nno_p1": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nno_p2": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nno_p3": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nno_p4": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nob": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nob_p0": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nob_p1": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nob_p2": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nob_p3": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_gen_nob_p4": f"{BASE_IMPORT}noreval_gen:NorevalGenerationExtractor",
    "nortruthfulqa_mc_nno": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nno_p0": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nno_p1": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nno_p2": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nno_p3": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nno_p4": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nob": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nob_p0": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nob_p1": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nob_p2": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nob_p3": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nortruthfulqa_mc_nob_p4": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
}
