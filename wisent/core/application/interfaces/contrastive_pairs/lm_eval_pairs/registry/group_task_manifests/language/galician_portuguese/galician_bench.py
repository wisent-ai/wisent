"""Galician bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

GALICIAN_BENCH_TASKS = {
    # Multiple_choice subtasks → galician_bench_mc.py (evaluator_name = "log_likelihoods")
    "belebele_glg_Latn": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "galcola": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "mgsm_direct_gl": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "openbookqa_gl": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "parafrases_gl": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "paws_gl": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "truthfulqa_gl_mc1": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "truthfulqa_gl_mc2": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "xnli_gl": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    "xstorycloze_gl": f"{BASE_IMPORT}galician_bench_mc:GalicianBenchMultipleChoiceExtractor",
    # Generate_until subtasks → galician_bench_gen.py (evaluator_name = "generation")
    "summarization_gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "truthfulqa_gl_gen": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_ca-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_de-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_en-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_es-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_eu-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_fr-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-ca": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-de": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-en": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-es": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-eu": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-fr": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-it": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl-pt": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_it-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
    "flores_pt-gl": f"{BASE_IMPORT}galician_bench_gen:GalicianBenchGenerationExtractor",
}
