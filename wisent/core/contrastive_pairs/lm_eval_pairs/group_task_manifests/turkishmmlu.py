"""TurkishMMLU group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

TURKISHMMLU_TASKS = {
    # Multiple_choice subtasks → turkishmmlu_mc.py (evaluator_name = "log_likelihoods")
    "turkishmmlu_biology": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_chemistry": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_geography": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_history": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_mathematics": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_philosophy": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_physics": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_religion_and_ethics": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    "turkishmmlu_turkish_language_and_literature": f"{BASE_IMPORT}turkishmmlu_mc:TurkishmmluMultipleChoiceExtractor",
    # Generate_until (CoT) subtasks → turkishmmlu_cot.py (evaluator_name = "generation")
    "turkishmmlu_cot_biology": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_chemistry": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_geography": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_history": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_mathematics": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_philosophy": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_physics": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_religion_and_ethics": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
    "turkishmmlu_cot_turkish_language_and_literature": f"{BASE_IMPORT}turkishmmlu_cot:TurkishmmluCotExtractor",
}
