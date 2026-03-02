"""Score group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

SCORE_TASKS = {
    "score_robustness": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_robustness_agieval": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_robustness_math": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_robustness_mmlu_pro": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_prompt_robustness_agieval": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_prompt_robustness_math": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_prompt_robustness_mmlu_pro": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_option_order_robustness_agieval": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_option_order_robustness_mmlu_pro": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_non_greedy_robustness_agieval": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_non_greedy_robustness_math": f"{BASE_IMPORT}score:ScoreExtractor",
    "score_non_greedy_robustness_mmlu_pro": f"{BASE_IMPORT}score:ScoreExtractor",
}
