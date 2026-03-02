"""Gpqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

GPQA_TASKS = {
    "gpqa": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_diamond_cot_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_diamond_cot_zeroshot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_diamond_generative_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_diamond_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_diamond_zeroshot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_extended_cot_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_extended_cot_zeroshot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_extended_generative_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_extended_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_extended_zeroshot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_main_cot_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_main_cot_zeroshot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_main_generative_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_main_n_shot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "gpqa_main_zeroshot": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "leaderboard_gpqa": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "leaderboard_gpqa_diamond": f"{BASE_IMPORT}gpqa:GPQAExtractor",
    "leaderboard_gpqa_extended": f"{BASE_IMPORT}gpqa:GPQAExtractor",
}
