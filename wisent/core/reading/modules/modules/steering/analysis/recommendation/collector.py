"""Ground truth types and collector."""
from __future__ import annotations
import json
import os
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from wisent.core.utils.config_tools.constants import DEFAULT_LIMIT, DEFAULT_N_TRIALS, JSON_INDENT, RECOMMEND_COLLECTOR_PER_TYPE, DEFAULT_RANDOM_SEED
from wisent.core.reading.modules.data.enriched_builder import build_enriched_from_db, generate_and_collect_enriched

# ── Data types ─────────────────────────────────────────────────

@dataclass(slots=True)
class MethodResult:
    """Accuracy of a single steering method on a single benchmark."""
    method: str
    accuracy: float
    layer: Optional[int] = None
    strength: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class BenchmarkGroundTruth:
    """One benchmark record: geometry metrics + per-method accuracy."""
    model: str
    benchmark: str
    metrics: Dict[str, Any]
    method_results: Dict[str, MethodResult]
    best_method: str
    best_accuracy: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["method_results"] = {
            k: asdict(v) for k, v in self.method_results.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkGroundTruth:
        mr = {k: MethodResult(**v)
              for k, v in d["method_results"].items()}
        return cls(
            model=d["model"], benchmark=d["benchmark"],
            metrics=d["metrics"], method_results=mr,
            best_method=d["best_method"],
            best_accuracy=d["best_accuracy"])


@dataclass(slots=True)
class GroundTruthDataset:
    """Collection of benchmark ground-truth records."""
    records: List[BenchmarkGroundTruth]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            [r.to_dict() for r in self.records], indent=JSON_INDENT))

    @classmethod
    def load(cls, path: str | Path) -> GroundTruthDataset:
        raw = json.loads(Path(path).read_text())
        return cls(records=[
            BenchmarkGroundTruth.from_dict(d) for d in raw])

    def __len__(self) -> int:
        return len(self.records)


# ── Collector ──────────────────────────────────────────────────
PIPELINE_METHODS = (
    "CAA", "Ostrze", "MLP", "TECZA",
    "TETNO", "GROM", "Concept Flow", "SZLAK", "WICHER")

# Method key mapping for Optuna objective
_METHOD_KEY_MAP = {
    "CAA": "CAA",
    "Ostrze": "OSTRZE",
    "MLP": "MLP",
    "TECZA": "TECZA",
    "TETNO": "TETNO",
    "GROM": "GROM",
    "Concept Flow": "NURT",
    "SZLAK": "SZLAK",
    "WICHER": "WICHER",
}


def _load_zwiad_metrics(model: str, benchmark: str, zwiad_dir: str) -> Optional[Dict]:
    slug, zd = model.replace("/", "_"), Path(zwiad_dir)
    path = zd / f"{slug}__{benchmark}.json"
    if not path.exists():
        hits = sorted(zd.glob(f"{slug}__*_{benchmark}.json"))
        path = hits[0] if hits else path
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("metrics", data)




def collect_benchmark_ground_truth(
    model_name: str, benchmark: str,
    zwiad_dir: str = "zwiad_results",
    limit: int = DEFAULT_LIMIT, device: Optional[str] = None,
    methods: Optional[List[str]] = None,
    n_trials: int = DEFAULT_N_TRIALS,
) -> Optional[BenchmarkGroundTruth]:
    """Run Optuna optimization for each method on one benchmark."""
    import optuna
    from wisent.core.utils.cli.optimize_steering import create_optuna_objective
    from wisent.core.primitives.models.wisent_model import WisentModel

    metrics = _load_zwiad_metrics(model_name, benchmark, zwiad_dir)
    if metrics is None:
        return None

    if methods is None:
        methods = list(PIPELINE_METHODS)

    # 1. Load model ONCE (needed for Optuna response generation)
    cached_model = WisentModel(model_name, device=device)
    num_layers = cached_model.num_layers
    print(f"  Loaded {model_name} ({num_layers} layers)")

    # 2. Build enriched pairs (try DB first, then generate from scratch)
    work_dir = tempfile.mkdtemp(prefix=f"gt_{benchmark}_")
    cfg = cached_model.hf_model.config
    enriched_file = build_enriched_from_db(
        model_name, benchmark, work_dir, limit=limit,
        num_attention_heads=getattr(cfg, 'num_attention_heads', None),
        num_key_value_heads=getattr(cfg, 'num_key_value_heads', None))
    if enriched_file is None:
        enriched_file = generate_and_collect_enriched(
            model_name, benchmark, work_dir, limit, device, cached_model)

    # 4. For each method: Optuna optimization
    method_results: Dict[str, MethodResult] = {}
    best_acc, best_method = -1.0, methods[0]

    for method in methods:
        if method not in PIPELINE_METHODS:
            print(f"  Skipping {method} (no pipeline support)")
            continue
        method_key = _METHOD_KEY_MAP.get(method, method.upper())
        print(f"  Optimizing {method} ({n_trials} trials)...")

        trial_work = os.path.join(work_dir, f"optuna_{method}")
        os.makedirs(trial_work, exist_ok=True)

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{benchmark}_{method}",
            sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_SEED),
        )

        objective = create_optuna_objective(
            model=model_name,
            task=benchmark,
            method=method_key,
            num_layers=num_layers,
            limit=limit,
            device=device,
            work_dir=trial_work,
            enriched_pairs_file=enriched_file,
            cached_model=cached_model,
        )

        try:
            study.optimize(
                objective, n_trials=n_trials,
                show_progress_bar=False)
            acc = study.best_value
            params = study.best_params
        except Exception as e:
            print(f"    {method} optimization failed: {e}")
            acc = 0.0
            params = {}

        layer = params.get(
            "layer", params.get("sensor_layer", None))
        method_results[method] = MethodResult(
            method=method, accuracy=acc,
            layer=layer, strength=params.get("max_alpha"),
            best_params=params,
        )
        if acc > best_acc:
            best_acc, best_method = acc, method
        print(f"    {method}: {acc:.4f} (best params: {params})")

    # 5. Free GPU memory
    del cached_model

    return BenchmarkGroundTruth(
        model=model_name, benchmark=benchmark,
        metrics=metrics, method_results=method_results,
        best_method=best_method, best_accuracy=best_acc)


def collect_ground_truth(
    model: str, benchmarks: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    zwiad_dir: str = "zwiad_results",
    limit: int = DEFAULT_LIMIT, device: Optional[str] = None,
    methods: Optional[List[str]] = None,
    n_trials: int = DEFAULT_N_TRIALS,
    benchmark_start: Optional[int] = None,
    benchmark_end: Optional[int] = None,
    use_geometry_selection: bool = False,
    per_type: int = RECOMMEND_COLLECTOR_PER_TYPE, fine_geometry: bool = False,
) -> GroundTruthDataset:
    """Collect ground truth. use_geometry_selection picks per_type per type."""
    if benchmarks is None:
        if use_geometry_selection:
            from wisent.core.reading.modules.zwiad.geometry_types import (
                select_representative_benchmarks, GeometryType, GeometryTypeFine)
            selected = select_representative_benchmarks(
                zwiad_dir, model, per_type=per_type, fine=fine_geometry)
            benchmarks = []
            enum_cls = GeometryTypeFine if fine_geometry else GeometryType
            for gtype in enum_cls:
                benches = selected.get(gtype, [])
                print(f"  {gtype.value}: {benches}")
                benchmarks.extend(benches)
        else:
            slug = model.replace("/", "_")
            rp = Path(zwiad_dir)
            benchmarks = sorted({
                f.stem.replace(f"{slug}__", "")
                for f in rp.glob(f"{slug}__*.json")})
    # Apply sharding
    if benchmark_start is not None or benchmark_end is not None:
        start = benchmark_start or 0
        end = benchmark_end or len(benchmarks)
        benchmarks = benchmarks[start:end]

    total = len(benchmarks)
    print(f"Collecting ground truth for {total} benchmarks "
          f"({n_trials} Optuna trials/method)")
    records = []
    for i, bench in enumerate(benchmarks):
        print(f"\n[{i+1}/{total}] {bench}")
        try:
            rec = collect_benchmark_ground_truth(
                model, bench, zwiad_dir, limit, device,
                methods, n_trials)
        except Exception as e:
            print(f"  Benchmark {bench} failed entirely: {e}")
            rec = None
        if rec is not None:
            records.append(rec)
            # Save incrementally after each benchmark
            if output_path:
                ds = GroundTruthDataset(records=list(records))
                ds.save(output_path)
                print(f"  Saved {len(records)} records so far")
    dataset = GroundTruthDataset(records=records)
    if output_path:
        dataset.save(output_path)
        print(f"\nSaved {len(records)} records to {output_path}")
    return dataset
