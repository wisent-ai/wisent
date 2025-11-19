"""Japanese leaderboard group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

JAPANESE_LEADERBOARD_TASKS = {
    "japanese_leaderboard": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_jaqket_v2": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_mgsm": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_xlsum": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_jsquad": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_jcommonsenseqa": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_jnli": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_marc_ja": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
    "ja_leaderboard_xwinograd": f"{BASE_IMPORT}japanese_leaderboard:JapaneseLeaderboardExtractor",
}
