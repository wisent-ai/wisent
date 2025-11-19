"""Arabic leaderboard complete group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

ARABIC_LEADERBOARD_COMPLETE_TASKS = {
    "arabic_leaderboard_complete": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_acva": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_alghafa": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_exams": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_exams": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_arc_challenge": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_arc_easy": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_boolq": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_hellaswag": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_mmlu": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_copa": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_openbook_qa": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_piqa": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_race": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_sciq": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
    "arabic_leaderboard_arabic_mt_toxigen": f"{BASE_IMPORT}arabic_leaderboard_complete:ArabicLeaderboardCompleteExtractor",
}
