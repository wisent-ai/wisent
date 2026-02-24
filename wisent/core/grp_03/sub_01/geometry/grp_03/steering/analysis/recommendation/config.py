"""Configurable recommendation parameters.

All thresholds and score weights live in RecommendationConfig.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from wisent.core.constants import (
    RECOMMEND_LINEAR_PROBE_HIGH, RECOMMEND_LINEAR_PROBE_LOW,
    RECOMMEND_NONLINEARITY_GAP, RECOMMEND_ICD_HIGH, RECOMMEND_ICD_LOW,
    RECOMMEND_STABILITY_HIGH, RECOMMEND_STABILITY_LOW,
    RECOMMEND_MULTI_CONCEPT_SILHOUETTE, RECOMMEND_ALIGNMENT_HIGH,
    RECOMMEND_ALIGNMENT_LOW, RECOMMEND_VARIANCE_CONCENTRATED,
    RECOMMEND_EFFECTIVE_DIMS_LOW, RECOMMEND_MULTI_DIR_GAIN_HIGH,
    RECOMMEND_COHERENCE_HIGH, RECOMMEND_COHERENCE_LOW,
    RECOMMEND_ICD_TOP5_THRESHOLD, RECOMMEND_NOT_VIABLE_SCALE,
    RECOMMEND_CONFIDENCE_BASE, SCORE_WEIGHT_DEFAULT,
)

METHODS = ("CAA", "Ostrze", "MLP", "TECZA", "TETNO", "GROM",
           "Concept Flow", "SZLAK", "WICHER")


@dataclass(slots=True)
class Thresholds:
    linear_probe_high: float = RECOMMEND_LINEAR_PROBE_HIGH
    linear_probe_low: float = RECOMMEND_LINEAR_PROBE_LOW
    nonlinearity_gap_significant: float = RECOMMEND_NONLINEARITY_GAP
    icd_high: float = RECOMMEND_ICD_HIGH
    icd_low: float = RECOMMEND_ICD_LOW
    stability_high: float = RECOMMEND_STABILITY_HIGH
    stability_low: float = RECOMMEND_STABILITY_LOW
    multi_concept_silhouette: float = RECOMMEND_MULTI_CONCEPT_SILHOUETTE
    alignment_high: float = RECOMMEND_ALIGNMENT_HIGH
    alignment_low: float = RECOMMEND_ALIGNMENT_LOW
    variance_concentrated: float = RECOMMEND_VARIANCE_CONCENTRATED
    effective_dims_low: float = RECOMMEND_EFFECTIVE_DIMS_LOW
    multi_dir_gain_high: float = RECOMMEND_MULTI_DIR_GAIN_HIGH
    coherence_high: float = RECOMMEND_COHERENCE_HIGH
    coherence_low: float = RECOMMEND_COHERENCE_LOW
    icd_top5_threshold: float = RECOMMEND_ICD_TOP5_THRESHOLD


@dataclass(slots=True)
class ScoreWeights:
    linear_high_caa: float = SCORE_WEIGHT_DEFAULT
    linear_high_ostrze: float = SCORE_WEIGHT_DEFAULT
    linear_high_cf: float = SCORE_WEIGHT_DEFAULT
    linear_low_mlp: float = SCORE_WEIGHT_DEFAULT
    linear_low_tetno: float = SCORE_WEIGHT_DEFAULT
    linear_low_grom: float = SCORE_WEIGHT_DEFAULT
    linear_low_cf: float = SCORE_WEIGHT_DEFAULT
    nonlin_gap_mlp: float = SCORE_WEIGHT_DEFAULT
    nonlin_gap_tetno: float = SCORE_WEIGHT_DEFAULT
    nonlin_gap_grom: float = SCORE_WEIGHT_DEFAULT
    nonlin_gap_cf: float = SCORE_WEIGHT_DEFAULT
    multi_concept_tecza: float = SCORE_WEIGHT_DEFAULT
    multi_concept_grom: float = SCORE_WEIGHT_DEFAULT
    multi_concept_cf: float = SCORE_WEIGHT_DEFAULT
    multi_concept_caa: float = SCORE_WEIGHT_DEFAULT
    multi_concept_ostrze: float = SCORE_WEIGHT_DEFAULT
    stab_high_caa: float = SCORE_WEIGHT_DEFAULT
    stab_high_ostrze: float = SCORE_WEIGHT_DEFAULT
    stab_high_cf: float = SCORE_WEIGHT_DEFAULT
    stab_low_grom: float = SCORE_WEIGHT_DEFAULT
    stab_low_tetno: float = SCORE_WEIGHT_DEFAULT
    stab_low_cf: float = SCORE_WEIGHT_DEFAULT
    stab_low_caa: float = SCORE_WEIGHT_DEFAULT
    icd_high_caa: float = SCORE_WEIGHT_DEFAULT
    icd_high_ostrze: float = SCORE_WEIGHT_DEFAULT
    icd_high_cf: float = SCORE_WEIGHT_DEFAULT
    icd_low_grom: float = SCORE_WEIGHT_DEFAULT
    icd_low_tetno: float = SCORE_WEIGHT_DEFAULT
    coh_high_caa: float = SCORE_WEIGHT_DEFAULT
    coh_low_tecza: float = SCORE_WEIGHT_DEFAULT
    coh_low_grom: float = SCORE_WEIGHT_DEFAULT
    coh_low_cf: float = SCORE_WEIGHT_DEFAULT
    var_pc1_cf: float = SCORE_WEIGHT_DEFAULT
    icd_top5_cf: float = SCORE_WEIGHT_DEFAULT
    mdir_gain_cf: float = SCORE_WEIGHT_DEFAULT
    mdir_gain_tecza: float = SCORE_WEIGHT_DEFAULT
    eff_dims_cf: float = SCORE_WEIGHT_DEFAULT
    linear_high_szlak: float = SCORE_WEIGHT_DEFAULT
    linear_low_szlak: float = SCORE_WEIGHT_DEFAULT
    nonlin_gap_szlak: float = SCORE_WEIGHT_DEFAULT
    multi_concept_szlak: float = SCORE_WEIGHT_DEFAULT
    stab_high_szlak: float = SCORE_WEIGHT_DEFAULT
    stab_low_szlak: float = SCORE_WEIGHT_DEFAULT
    icd_high_szlak: float = SCORE_WEIGHT_DEFAULT
    icd_low_szlak: float = SCORE_WEIGHT_DEFAULT
    coh_low_szlak: float = SCORE_WEIGHT_DEFAULT
    var_pc1_szlak: float = SCORE_WEIGHT_DEFAULT
    mdir_gain_szlak: float = SCORE_WEIGHT_DEFAULT
    eff_dims_szlak: float = SCORE_WEIGHT_DEFAULT
    linear_high_wicher: float = SCORE_WEIGHT_DEFAULT
    linear_low_wicher: float = SCORE_WEIGHT_DEFAULT
    nonlin_gap_wicher: float = SCORE_WEIGHT_DEFAULT
    multi_concept_wicher: float = SCORE_WEIGHT_DEFAULT
    stab_high_wicher: float = SCORE_WEIGHT_DEFAULT
    stab_low_wicher: float = SCORE_WEIGHT_DEFAULT
    icd_high_wicher: float = SCORE_WEIGHT_DEFAULT
    icd_low_wicher: float = SCORE_WEIGHT_DEFAULT
    coh_high_wicher: float = SCORE_WEIGHT_DEFAULT
    coh_low_wicher: float = SCORE_WEIGHT_DEFAULT
    not_viable_scale: float = RECOMMEND_NOT_VIABLE_SCALE
    confidence_base: float = RECOMMEND_CONFIDENCE_BASE


@dataclass(slots=True)
class RecommendationConfig:
    thresholds: Thresholds
    weights: ScoreWeights

    @staticmethod
    def default() -> RecommendationConfig:
        return RecommendationConfig(
            thresholds=Thresholds(), weights=ScoreWeights())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RecommendationConfig:
        data = json.loads(Path(path).read_text())
        return cls(thresholds=Thresholds(**data["thresholds"]),
                   weights=ScoreWeights(**data["weights"]))

    def param_count(self) -> int:
        return len(fields(self.thresholds)) + len(fields(self.weights))
