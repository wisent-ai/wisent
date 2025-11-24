"""Portuguese bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

PORTUGUESE_BENCH_TASKS = {
    # Multiple_choice subtasks → portuguese_bench_mc.py (evaluator_name = "log_likelihoods")
    "assin_entailment": f"{BASE_IMPORT}portuguese_bench_mc:PortugueseBenchMultipleChoiceExtractor",
    "assin_paraphrase": f"{BASE_IMPORT}portuguese_bench_mc:PortugueseBenchMultipleChoiceExtractor",
    # Generate_until subtasks → portuguese_bench_gen.py (evaluator_name = "generation")
    "flores_ca-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_de-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_en-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_es-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_eu-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_fr-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_gl-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_it-pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-ca": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-de": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-en": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-es": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-eu": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-fr": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-gl": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt-it": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
    "flores_pt": f"{BASE_IMPORT}portuguese_bench_gen:PortugueseBenchGenerationExtractor",
}
