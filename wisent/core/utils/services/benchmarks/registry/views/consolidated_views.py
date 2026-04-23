"""Filter API over tasks_consolidated.json.

Migration target. Once every consumer imports from here, the 16 dead snapshot
JSONs under wisent/support/parameters/lm_eval/ and wisent/support/parameters/
can be deleted without behavior change.

Cache the parsed registry in module state so each filter pays one JSON parse
per process, not per call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


def _registry_path() -> Path:
    this_dir = Path(__file__).resolve().parent
    wisent_root = this_dir.parent.parent.parent.parent.parent.parent
    return wisent_root / "support" / "parameters" / "tasks" / "tasks_consolidated.json"


_cache: dict | None = None
_alias_to_canon: dict[str, str] | None = None


def _canon(name: str) -> str:
    return name.lower().replace("-", "_").replace("/", "_")


def _load() -> dict:
    global _cache, _alias_to_canon
    if _cache is None:
        _cache = json.loads(_registry_path().read_text())
        _alias_to_canon = {}
        for canon_name, row in _cache["tasks"].items():
            _alias_to_canon[canon_name] = canon_name
            for alias in row.get("aliases", []):
                _alias_to_canon[_canon(alias)] = canon_name
    return _cache


def _tasks() -> dict[str, dict]:
    return _load()["tasks"]


def resolve_name(name: str) -> Optional[str]:
    _load()
    return _alias_to_canon.get(_canon(name))


def get_task(name: str) -> Optional[dict]:
    canon = resolve_name(name)
    return _tasks().get(canon) if canon else None


def get_working_tasks() -> list[str]:
    return sorted(c for c, row in _tasks().items() if row.get("is_in_working_list"))


def get_broken_tasks(blocker: Optional[str] = None) -> list[str]:
    out = []
    for c, row in _tasks().items():
        b = row.get("blocker")
        if not b:
            continue
        if blocker and b != blocker:
            continue
        out.append(c)
    return sorted(out)


def get_unfixable_tasks() -> list[str]:
    return sorted(c for c, row in _tasks().items() if row.get("status") == "unfixable")


def get_tasks_by_category(category: str) -> list[str]:
    return sorted(c for c, row in _tasks().items() if row.get("category") == category)


def get_all_categories() -> list[str]:
    return sorted({row["category"] for row in _tasks().values() if row.get("category")})


def get_tasks_by_source(source: str) -> list[str]:
    return sorted(c for c, row in _tasks().items() if row.get("source") == source)


def get_tasks_by_kind(kind: str) -> list[str]:
    return sorted(c for c, row in _tasks().items() if row.get("kind") == kind)


def get_tasks_by_skill(skill: str) -> list[str]:
    return sorted(c for c, row in _tasks().items() if skill in row.get("skills", []))


def get_tasks_by_risk(risk: str) -> list[str]:
    return sorted(c for c, row in _tasks().items() if risk in row.get("risks", []))


def get_tasks_with_pair_texts() -> list[str]:
    return sorted(c for c, row in _tasks().items() if row.get("has_pair_texts_on_hf"))


def get_tasks_with_activations(
    model: str, strategy: Optional[str] = None,
) -> list[str]:
    out = []
    for c, row in _tasks().items():
        cov = row.get("activations_coverage") or {}
        if model not in cov:
            continue
        if strategy and strategy not in cov[model]:
            continue
        out.append(c)
    return sorted(out)


def get_tasks_missing_activations(
    model: str, strategy: Optional[str] = None,
    only_working: bool = True,
) -> list[str]:
    out = []
    for c, row in _tasks().items():
        if only_working and not row.get("is_in_working_list"):
            continue
        cov = row.get("activations_coverage") or {}
        strats = cov.get(model, [])
        if not strats or (strategy and strategy not in strats):
            out.append(c)
    return sorted(out)


def get_all_benchmarks(exclude_broken: bool = True) -> list[str]:
    out = []
    for c, row in _tasks().items():
        if exclude_broken and row.get("status") in {"broken", "unfixable"}:
            continue
        out.append(c)
    return sorted(out)


def get_huggingface_only_tasks() -> list[str]:
    return get_tasks_by_source("hf_only")


def get_lm_eval_tasks() -> list[str]:
    return get_tasks_by_source("lm_eval")


def find_tasks_by_tags(
    *, skills: Iterable[str] | None = None,
    risks: Iterable[str] | None = None,
    min_quality_score: int = 2,
) -> list[str]:
    required_tags: set[str] = set()
    if skills:
        required_tags.update(skills)
    if risks:
        required_tags.update(risks)
    out = []
    for c, row in _tasks().items():
        if (row.get("quality_score") or 0) < min_quality_score:
            continue
        task_tags = set(row.get("tags", []))
        if required_tags and not (task_tags & required_tags):
            continue
        out.append(c)
    return sorted(out)


def meta() -> dict:
    return _load()["_meta"]
