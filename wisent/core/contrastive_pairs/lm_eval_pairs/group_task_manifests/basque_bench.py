"""Basque bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

BASQUE_BENCH_TASKS = {
    # Multiple_choice subtasks → basque_bench_mc.py (evaluator_name = "log_likelihoods")
    "arc_eu_easy": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    "arc_eu_challenge": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    "paws_eu": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    "piqa_eu": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    "wnli_eu": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    "xcopa_eu": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    "mgsm_direct_eu": f"{BASE_IMPORT}basque_bench_mc:BasqueBenchMultipleChoiceExtractor",
    # Generate_until subtasks → basque_bench_gen.py (evaluator_name = "generation")
    "mgsm_native_cot_eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_ca-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_de-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_en-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_es-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-ca": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-de": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-en": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-es": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-fr": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-gl": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-it": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu-pt": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_fr-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_gl-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_it-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
    "flores_pt-eu": f"{BASE_IMPORT}basque_bench_gen:BasqueBenchGenerationExtractor",
}
