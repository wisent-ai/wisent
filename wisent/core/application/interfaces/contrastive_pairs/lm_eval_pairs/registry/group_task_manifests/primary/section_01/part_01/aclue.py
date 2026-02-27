"""Aclue group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

ACLUE_TASKS = {
    "aclue": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_ancient_chinese_culture": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_ancient_literature": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_ancient_medical": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_ancient_phonetics": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_basic_ancient_chinese": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_couplet_prediction": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_homographic_character_resolution": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_named_entity_recognition": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_poetry_appreciate": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_poetry_context_prediction": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_poetry_quality_assessment": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_poetry_sentiment_analysis": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_polysemy_resolution": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_reading_comprehension": f"{BASE_IMPORT}aclue:AclueExtractor",
    "aclue_sentence_segmentation": f"{BASE_IMPORT}aclue:AclueExtractor",
}
