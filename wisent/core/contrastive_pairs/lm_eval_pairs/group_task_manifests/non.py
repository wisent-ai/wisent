"""Non group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

NON_TASKS = {
    "non_greedy_robustness_agieval_aqua_rat": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_agieval_logiqa_en": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_agieval_lsat_ar": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_agieval_lsat_lr": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_agieval_lsat_rc": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_agieval_sat_en": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_agieval_sat_math": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_algebra": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_counting_and_prob": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_geometry": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_intermediate_algebra": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_num_theory": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_prealgebra": f"{BASE_IMPORT}score:ScoreExtractor",
    "non_greedy_robustness_math_precalc": f"{BASE_IMPORT}score:ScoreExtractor",
    "non": f"{BASE_IMPORT}non:NonExtractor",
}
