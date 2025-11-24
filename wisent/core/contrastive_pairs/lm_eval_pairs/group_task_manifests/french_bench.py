"""French bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

FRENCH_BENCH_TASKS = {
    # Multiple_choice subtasks → french_bench_mc.py (evaluator_name = "log_likelihoods")
    "french_bench_arc_challenge": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_boolqa": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_hellaswag": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_multifquad": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_reading_comp": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_topic_based_nli": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_trivia": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_vocab": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    "french_bench_xnli": f"{BASE_IMPORT}french_bench_mc:FrenchBenchMultipleChoiceExtractor",
    # Loglikelihood_rolling subtasks → french_bench_perplexity.py (evaluator_name = "log_likelihoods")
    "french_bench_opus_perplexity": f"{BASE_IMPORT}french_bench_perplexity:FrenchBenchPerplexityExtractor",
    "french_bench_wikitext_fr": f"{BASE_IMPORT}french_bench_perplexity:FrenchBenchPerplexityExtractor",
}
