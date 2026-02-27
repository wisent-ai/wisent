"""Paloma group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

PALOMA_TASKS = {
    "paloma": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_4chan_meta_sep": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_c4_100_domains": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_c4_en": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_dolma-v1_5": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_dolma_100_programing_languages": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_dolma_100_subreddits": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_falcon-refinedweb": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_gab": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_m2d2_s2orc_unsplit": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_m2d2_wikipedia_unsplit": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_manosphere_meta_sep": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_mc4": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_ptb": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_redpajama": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_twitterAAE_HELM_fixed": f"{BASE_IMPORT}paloma:PalomaExtractor",
    "paloma_wikitext_103": f"{BASE_IMPORT}paloma:PalomaExtractor",
}
