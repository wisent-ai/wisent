"""Catalan bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

CATALAN_BENCH_TASKS = {
    # Multiple_choice subtasks → catalan_bench_mc.py (evaluator_name = "log_likelihoods")
    "arc_ca_easy": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "arc_ca_challenge": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "catalanqa": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "catcola": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "cocoteros_va": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "copa_ca": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "coqcat": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "mgsm_direct_ca": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "openbookqa_ca": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "parafraseja": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    "paws_ca": f"{BASE_IMPORT}catalan_bench_mc:CatalanBenchMultipleChoiceExtractor",
    # Generate_until subtasks → catalan_bench_gen.py (evaluator_name = "generation")
    "cabreu_abstractive": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "cabreu_extractive": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "cabreu_extreme": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-de": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-en": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-es": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-eu": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-fr": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-gl": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-it": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca-pt": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_de-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_en-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_es-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_eu-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_fr-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_gl-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_it-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
    "flores_pt-ca": f"{BASE_IMPORT}catalan_bench_gen:CatalanBenchGenerationExtractor",
}
