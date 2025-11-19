"""Pile group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

PILE_TASKS = {
    "pile": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_arxiv": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_bookcorpus2": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_books3": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_dm-mathematics": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_enron": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_europarl": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_freelaw": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_github": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_gutenberg": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_hackernews": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_nih-exporter": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_opensubtitles": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_openwebtext2": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_philpapers": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_pile-cc": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_pubmed-abstracts": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_pubmed-central": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_stackexchange": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_ubuntu-irc": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_uspto": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_wikipedia": f"{BASE_IMPORT}pile:PileExtractor",
    "pile_youtubesubtitles": f"{BASE_IMPORT}pile:PileExtractor",
}
