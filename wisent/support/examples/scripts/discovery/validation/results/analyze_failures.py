"""Analyze and clean failing benchmarks grouped by error detail.

Usage:
    python analyze_failures.py          # show failure groups
    python analyze_failures.py clean-stale  # delete stale cache entries
    python analyze_failures.py test-import  # test if wisent imports work
"""
import json
import os
import sys

cache_dir = os.path.join(os.path.dirname(__file__), "hf_cache", "test_results")

mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"

failures = {}
for fname in sorted(os.listdir(cache_dir)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(cache_dir, fname)) as f:
        data = json.load(f)
    ext = data.get("extraction", {})
    if ext.get("status") != "PASS":
        detail = ext.get("detail", "no detail")
        if len(detail) > 150:
            detail = detail[:150]
        failures[fname[:-5]] = detail

if mode == "clean-stale":
    # Delete all remaining failure cache files (all are stale)
    count = 0
    for name in failures:
        path = os.path.join(cache_dir, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            count += 1
    print(f"Deleted {count} stale cache files")
elif mode == "extractor-for":
    task_name = sys.argv[2]
    from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor
    try:
        ext = get_extractor(task_name)
        print(f"Task '{task_name}' uses extractor: {type(ext).__name__}")
        print(f"  Module: {type(ext).__module__}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    sys.exit(0)

if mode == "inspect-task-choices":
    task_name = sys.argv[2]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".." * 7))
    from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
    task_data = LMEvalDataLoader.load_lm_eval_task(task_name)
    if isinstance(task_data, dict):
        first_key = list(task_data.keys())[0]
        task_obj = task_data[first_key]
    else:
        task_obj = task_data
    print(f"Task type: {type(task_obj).__name__}")
    config = getattr(task_obj, "config", None)
    if config:
        d2c = getattr(config, "doc_to_choice", None)
        print(f"config.doc_to_choice type: {type(d2c).__name__}")
        print(f"config.doc_to_choice value: {d2c}")
    if hasattr(task_obj, "doc_to_choice"):
        print(f"task.doc_to_choice exists, callable: {callable(task_obj.doc_to_choice)}")
    docs = []
    for method in ['test_docs', 'validation_docs', 'training_docs']:
        if hasattr(task_obj, method):
            try:
                result = getattr(task_obj, method)()
                if result is not None:
                    docs = list(result)
                    break
            except Exception:
                pass
    if docs:
        print(f"\nFirst doc keys: {list(docs[0].keys())}")
        for k, v in docs[0].items():
            v_str = str(v)
            if len(v_str) > 100:
                v_str = v_str[:100] + "..."
            print(f"  {k}: {v_str}")
    sys.exit(0)

if mode == "categorize-failing":
    progress_file = os.path.join(os.path.dirname(__file__), "refresh_progress.json")
    with open(progress_file) as f:
        prog = json.load(f)
    still = prog.get("still_failing", [])
    broken_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                                "support", "parameters", "lm_eval", "broken_in_lm_eval.json")
    with open(broken_path) as f:
        broken = set(json.load(f))
    in_broken = [n for n in still if n in broken]
    not_in_broken = [n for n in still if n not in broken]
    print(f"Still failing: {len(still)}")
    print(f"  Of which intentionally broken: {len(in_broken)}")
    print(f"  Actually need fixing: {len(not_in_broken)}")
    print(f"\nNot in broken list (need fixing):")
    for n in sorted(not_in_broken):
        print(f"  {n}")
    sys.exit(0)

if mode == "find-group":
    task = sys.argv[2]
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import GROUP_TASK_EXPANSIONS
    def _normalize(name):
        return name.strip().lower().replace("-", "_")
    norm = _normalize(task)
    for k, v in GROUP_TASK_EXPANSIONS.items():
        if _normalize(k) == norm:
            print(f"Match: '{k}' -> {len(v)} subtasks: {v[:5]}")
    sys.exit(0)

if mode == "trace-load":
    task_name = sys.argv[2]
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import TASK_NAME_MAPPING
    mapped = TASK_NAME_MAPPING.get(task_name, task_name)
    print(f"Original: {task_name}")
    print(f"After TASK_NAME_MAPPING: {mapped}")
    import re as _re
    _CASE_PREFIX_MAP = {"aradice_": "AraDiCE_", "tinybenchmarks_": "tinyBenchmarks_"}
    for lp, cp in _CASE_PREFIX_MAP.items():
        if mapped.startswith(lp):
            mapped = cp + mapped[len(lp):]
            break
    print(f"After case prefix: {mapped}")
    _ISO_SCRIPTS = {
        "latn": "Latn", "cyrl": "Cyrl", "arab": "Arab", "ethi": "Ethi",
        "deva": "Deva", "hebr": "Hebr", "nkoo": "Nkoo", "beng": "Beng",
        "guru": "Guru", "mlym": "Mlym", "taml": "Taml", "orya": "Orya",
        "sinh": "Sinh", "mymr": "Mymr", "khmr": "Khmr", "hang": "Hang",
        "laoo": "Laoo", "tibt": "Tibt", "grek": "Grek", "armn": "Armn",
        "jpan": "Jpan", "knda": "Knda", "geor": "Geor", "telu": "Telu",
        "thai": "Thai", "hans": "Hans", "hant": "Hant", "gujr": "Gujr",
    }
    for lower, title in _ISO_SCRIPTS.items():
        mapped = _re.sub(rf'(?<=[_\-]){lower}(?=[_\-]|$)', title, mapped)
    print(f"After ISO restore: {mapped}")
    sys.exit(0)

if mode == "show-still-failing":
    progress_file = os.path.join(os.path.dirname(__file__), "refresh_progress.json")
    with open(progress_file) as f:
        prog = json.load(f)
    still = prog.get("still_failing", [])
    print(f"Still failing ({len(still)}):")
    for n in still:
        print(f"  {n}")
    sys.exit(0)

if mode == "test-import-module":
    module_path = sys.argv[2]
    try:
        import importlib
        m = importlib.import_module(module_path)
        print(f"OK: imported {module_path}")
        print(f"  Has classes: {[n for n in dir(m) if not n.startswith('_')][:10]}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    sys.exit(0)

if mode == "test-import":
    print("Testing wisent import chain...")
    try:
        from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
        print(f"OK: _REGISTRY has {len(_REGISTRY)} entries")
    except Exception:
        import traceback
        traceback.print_exc()

if mode == "inspect-docs":
    # Inspect the raw lm-eval docs for a given task
    task_name = sys.argv[2] if len(sys.argv) > 2 else "aradice_egypt_cultural"
    print(f"Inspecting docs for: {task_name}")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".." * 7))
    from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
    task_data = LMEvalDataLoader.load_lm_eval_task(task_name)
    if isinstance(task_data, dict):
        print(f"Group task with {len(task_data)} subtasks: {list(task_data.keys())[:5]}")
        first_key = list(task_data.keys())[0]
        task_obj = task_data[first_key]
    else:
        task_obj = task_data
    print(f"Task type: {type(task_obj).__name__}")
    print(f"Task NAME: {getattr(task_obj, 'NAME', 'unknown')}")
    docs = []
    for method in ['test_docs', 'validation_docs', 'training_docs']:
        if hasattr(task_obj, method):
            result = getattr(task_obj, method)()
            if result is not None:
                docs = list(result)
                print(f"Loaded docs from {method}: {len(docs)}")
                break
    print(f"Doc count: {len(docs)}")
    if docs:
        print(f"\nFirst doc keys: {list(docs[0].keys())}")
        import json
        for k, v in docs[0].items():
            v_str = str(v)
            if len(v_str) > 100:
                v_str = v_str[:100] + f"... ({type(v).__name__}, len={len(v_str)})"
            print(f"  {k}: {v_str}")
else:
    detail_groups = {}
    for name, detail in failures.items():
        detail_groups.setdefault(detail, []).append(name)

    for detail, names in sorted(detail_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n[{len(names)} benchmarks] {detail}")
        for n in names:
            print(f"  - {n}")

    print(f"\nTotal failing: {len(failures)}")
    print(f"Distinct error groups: {len(detail_groups)}")
