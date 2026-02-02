"""Constants for benchmark processing."""

from __future__ import annotations

# Path to lm_eval tasks directory
LM_EVAL_TASKS_PATH = (
    "/opt/homebrew/Caskroom/miniforge/base/lib/python3.10/site-packages/lm_eval/tasks"
)

# Approved tags from skills.json and risks.json
APPROVED_SKILLS = [
    "coding",
    "mathematics",
    "long context",
    "creative writing",
    "general knowledge",
    "medical",
    "law",
    "science",
    "history",
    "tool use",
    "multilingual",
    "reasoning",
]

APPROVED_RISKS = [
    "harmfulness",
    "toxicity",
    "bias",
    "hallucination",
    "violence",
    "adversarial robustness",
    "sycophancy",
    "deception",
]

__all__ = ["LM_EVAL_TASKS_PATH", "APPROVED_SKILLS", "APPROVED_RISKS"]
