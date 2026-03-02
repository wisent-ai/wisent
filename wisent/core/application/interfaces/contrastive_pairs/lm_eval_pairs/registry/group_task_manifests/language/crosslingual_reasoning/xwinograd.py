"""Xwinograd group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

XWINOGRAD_TASKS = {
    "xwinograd": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
    "xwinograd_en": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
    "xwinograd_fr": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
    "xwinograd_jp": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
    "xwinograd_pt": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
    "xwinograd_ru": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
    "xwinograd_zh": f"{BASE_IMPORT}xwinograd:XWinogradExtractor",
}
