"""Unitxt group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

UNITXT_TASKS = {
    "20_newsgroups": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "ag_news": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "argument_topic": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "banking77": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "claim_stance_topic": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "cnn_dailymail": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "dbpedia_14": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "ethos_binary": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "financial_tweets": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "law_stack_exchange": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "ledgar": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "medical_abstracts": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "unfair_tos": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "xsum": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
    "yahoo_answers_topics": f"{BASE_IMPORT}unitxt:UnitxtExtractor",
}
