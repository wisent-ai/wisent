"""Japanese leaderboard group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

JAPANESE_LEADERBOARD_TASKS = {
    # Generate_until subtasks → japanese_leaderboard_gen.py (evaluator_name = "generation")
    "ja_leaderboard_jaqket_v2": f"{BASE_IMPORT}japanese_leaderboard_gen:JapaneseLeaderboardGenerationExtractor",
    "ja_leaderboard_jsquad": f"{BASE_IMPORT}japanese_leaderboard_gen:JapaneseLeaderboardGenerationExtractor",
    "ja_leaderboard_mgsm": f"{BASE_IMPORT}japanese_leaderboard_gen:JapaneseLeaderboardGenerationExtractor",
    "ja_leaderboard_xlsum": f"{BASE_IMPORT}japanese_leaderboard_gen:JapaneseLeaderboardGenerationExtractor",
    # Multiple_choice subtasks → japanese_leaderboard_mc.py (evaluator_name = "log_likelihoods")
    "ja_leaderboard_jcommonsenseqa": f"{BASE_IMPORT}japanese_leaderboard_mc:JapaneseLeaderboardMultipleChoiceExtractor",
    "ja_leaderboard_jnli": f"{BASE_IMPORT}japanese_leaderboard_mc:JapaneseLeaderboardMultipleChoiceExtractor",
    "ja_leaderboard_marc_ja": f"{BASE_IMPORT}japanese_leaderboard_mc:JapaneseLeaderboardMultipleChoiceExtractor",
    "ja_leaderboard_xwinograd": f"{BASE_IMPORT}japanese_leaderboard_mc:JapaneseLeaderboardMultipleChoiceExtractor",
}
