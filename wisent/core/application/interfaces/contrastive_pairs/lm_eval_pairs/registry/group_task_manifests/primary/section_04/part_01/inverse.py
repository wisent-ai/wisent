"""Inverse group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

INVERSE_TASKS = {
    "inverse_scaling": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_hindsight_neglect_10shot": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_into_the_unknown": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_mc": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_memo_trap": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_modus_tollens": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_neqa": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_pattern_matching_suppression": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_quote_repetition": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_redefine_math": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_repetitive_algebra": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_sig_figs": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse_scaling_winobias_antistereotype": f"{BASE_IMPORT}inverse_scaling:InverseScalingExtractor",
    "inverse": f"{BASE_IMPORT}inverse:InverseExtractor",
}
