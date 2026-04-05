"""Advanced AI Risk group task manifest."""

from __future__ import annotations

from wisent.core.utils.config_tools.constants import LM_EVAL_EXTRACTOR_BASE_IMPORT

BASE_IMPORT = LM_EVAL_EXTRACTOR_BASE_IMPORT

# All 50 Advanced AI Risk tasks
# Note: fewshot, human, and lm variants are part of model_written_evals benchmark and use ModelWrittenEvalsExtractor
ADVANCED_AI_RISK_TASKS = {
    "advanced_ai_risk": f"{BASE_IMPORT}advanced:AdvancedExtractor",
    "advanced_ai_risk_fewshot_coordinate_itself": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_coordinate_other_ais": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_coordinate_other_versions": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_corrigible_less_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_corrigible_more_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_corrigible_neutral_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_myopic_reward": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_one_box_tendency": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_power_seeking_inclination": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_self_awareness_general_ai": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_self_awareness_good_text_model": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_self_awareness_text_model": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_self_awareness_training_architecture": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_self_awareness_training_web_gpt": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_survival_instinct": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_fewshot_wealth_seeking_inclination": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-coordinate_itself": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-coordinate_other_ais": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-coordinate_other_versions": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-corrigible_less_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-corrigible_more_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-corrigible_neutral_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-myopic_reward": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-one_box_tendency": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-power_seeking_inclination": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-self_awareness_general_ai": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-self_awareness_good_text_model": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-self_awareness_text_model": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-self_awareness-training_architecture": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-self_awareness_web_gpt": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-survival_instinct": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_human-wealth_seeking_inclination": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-coordinate_itself": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-coordinate_other_ais": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-coordinate_other_versions": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-corrigible_less_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-corrigible_more_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-corrigible_neutral_HHH": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-myopic_reward": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-one_box_tendency": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-power_seeking_inclination": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-self_awareness_general_ai": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-self_awareness_good_text_model": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-self_awareness_text_model": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-self_awareness-training_architecture": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-self_awareness-training_nn_architecture": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-self_awareness-training_web_gpt": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-survival_instinct": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    "advanced_ai_risk_lm-wealth_seeking_inclination": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
}
