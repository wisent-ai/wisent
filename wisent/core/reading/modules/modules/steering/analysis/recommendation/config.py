"""Configurable recommendation parameters.

All thresholds and score weights live in RecommendationConfig.
All fields are required: callers must provide explicit values.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from wisent.core.utils.config_tools.constants import JSON_INDENT

METHODS = ("CAA", "Ostrze", "MLP", "TECZA", "TETNO", "GROM",
           "Concept Flow", "SZLAK", "WICHER")


@dataclass(slots=True)
class Thresholds:
    linear_probe_high: float
    linear_probe_low: float
    nonlinearity_gap_significant: float
    icd_high: float
    icd_low: float
    stability_high: float
    stability_low: float
    multi_concept_silhouette: float
    alignment_high: float
    alignment_low: float
    variance_concentrated: float
    effective_dims_low: float
    multi_dir_gain_high: float
    coherence_high: float
    coherence_low: float
    icd_top5_threshold: float


@dataclass(slots=True)
class ScoreWeights:
    not_viable_scale: float
    confidence_base: float
    linear_high_caa: float
    linear_high_ostrze: float
    linear_high_cf: float
    linear_low_mlp: float
    linear_low_tetno: float
    linear_low_grom: float
    linear_low_cf: float
    nonlin_gap_mlp: float
    nonlin_gap_tetno: float
    nonlin_gap_grom: float
    nonlin_gap_cf: float
    multi_concept_tecza: float
    multi_concept_grom: float
    multi_concept_cf: float
    multi_concept_caa: float
    multi_concept_ostrze: float
    stab_high_caa: float
    stab_high_ostrze: float
    stab_high_cf: float
    stab_low_grom: float
    stab_low_tetno: float
    stab_low_cf: float
    stab_low_caa: float
    icd_high_caa: float
    icd_high_ostrze: float
    icd_high_cf: float
    icd_low_grom: float
    icd_low_tetno: float
    coh_high_caa: float
    coh_low_tecza: float
    coh_low_grom: float
    coh_low_cf: float
    var_pc1_cf: float
    icd_top5_cf: float
    mdir_gain_cf: float
    mdir_gain_tecza: float
    eff_dims_cf: float
    linear_high_szlak: float
    linear_low_szlak: float
    nonlin_gap_szlak: float
    multi_concept_szlak: float
    stab_high_szlak: float
    stab_low_szlak: float
    icd_high_szlak: float
    icd_low_szlak: float
    coh_low_szlak: float
    var_pc1_szlak: float
    mdir_gain_szlak: float
    eff_dims_szlak: float
    linear_high_wicher: float
    linear_low_wicher: float
    nonlin_gap_wicher: float
    multi_concept_wicher: float
    stab_high_wicher: float
    stab_low_wicher: float
    icd_high_wicher: float
    icd_low_wicher: float
    coh_high_wicher: float
    coh_low_wicher: float


@dataclass(slots=True)
class RecommendationConfig:
    thresholds: Thresholds
    weights: ScoreWeights

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=JSON_INDENT))

    @classmethod
    def load(cls, path: str | Path) -> RecommendationConfig:
        data = json.loads(Path(path).read_text())
        return cls(thresholds=Thresholds(**data["thresholds"]),
                   weights=ScoreWeights(**data["weights"]))

    def param_count(self) -> int:
        return len(fields(self.thresholds)) + len(fields(self.weights))
