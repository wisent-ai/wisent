"""Spanish bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

SPANISH_BENCH_TASKS = {
    # Multiple_choice subtasks → spanish_bench_mc.py (evaluator_name = "log_likelihoods")
    "cocoteros_es": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "copa_es": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "escola": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "mgsm_direct_es_spanish_bench": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "openbookqa_es": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "paws_es_spanish_bench": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "wnli_es": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    "xnli_es_spanish_bench": f"{BASE_IMPORT}spanish_bench_mc:SpanishBenchMultipleChoiceExtractor",
    # Generate_until subtasks → spanish_bench_gen.py (evaluator_name = "generation")
    "xlsum_es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_ca-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_de-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_en-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-ca": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-de": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-en": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-eu": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-fr": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-gl": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-it": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es-pt": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_eu-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_fr-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_gl-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_it-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
    "flores_pt-es": f"{BASE_IMPORT}spanish_bench_gen:SpanishBenchGenerationExtractor",
}
