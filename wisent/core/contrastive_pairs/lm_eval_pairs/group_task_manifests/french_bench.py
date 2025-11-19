"""French bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

FRENCH_BENCH_TASKS = {
    "french_bench": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_arc_challenge": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_boolqa": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_extra": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_fquadv2": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_fquadv2_bool": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_fquadv2_genq": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_fquadv2_hasAns": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_gen": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_grammar": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_hellaswag": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_mc": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_multifquad": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_opus_perplexity": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_orangesum_abstract": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_orangesum_title": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_perplexity": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_reading_comp": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_topic_based_nli": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_trivia": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_vocab": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_wikitext_fr": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
    "french_bench_xnli": f"{BASE_IMPORT}french_bench:FrenchBenchExtractor",
}
