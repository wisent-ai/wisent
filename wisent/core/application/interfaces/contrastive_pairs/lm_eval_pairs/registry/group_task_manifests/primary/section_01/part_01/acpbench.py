"""Acpbench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

ACPBENCH_TASKS = {
    "acp_areach_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_app_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_just_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_land_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_prog_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_reach_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_val_bool": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_areach_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_app_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_just_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_land_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_prog_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_reach_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_val_mcq": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
    "acp_areach_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_app_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_just_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_land_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_nexta_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_prog_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_reach_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_val_gen": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_areach_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_app_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_just_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_land_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_nexta_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_prog_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_reach_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acp_val_gen_with_pddl": f"{BASE_IMPORT}acp_bench_hard:AcpBenchHardExtractor",
    "acpbench": f"{BASE_IMPORT}acp_bench:AcpBenchExtractor",
}
