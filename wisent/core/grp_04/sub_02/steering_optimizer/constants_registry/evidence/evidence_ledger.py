"""Evidence ledger: persist, query, and reduce axis evidence records."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .evidence_data import (
    AxisEvidence, AxisReduction,
    CROSS_MODEL_CONFIDENCE_DECAY, MAX_AGE_DAYS, _model_family,
)

logger = logging.getLogger(__name__)
_LEDGER_DIR_NAME = "evidence_ledger"
_GCS_PREFIX = "evidence_ledger"


class EvidenceLedger:
    """Manage evidence records on disk and compute search-space reductions."""

    def __init__(self, ledger_dir: Optional[Path] = None):
        if ledger_dir is None:
            self._dir = Path.home() / ".wisent" / _LEDGER_DIR_NAME
        else:
            self._dir = ledger_dir

    @property
    def ledger_dir(self) -> Path:
        return self._dir

    # ---- Persistence ----------------------------------------------------

    def record(self, evidence: AxisEvidence) -> Path:
        """Save a single evidence record as JSON."""
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_model = evidence.model_name.replace("/", "_")
        fname = (
            f"{evidence.axis_name}_{safe_model}_"
            f"{evidence.task_name}_{evidence.method_name}_"
            f"{evidence.id}.json"
        )
        path = self._dir / fname
        with open(path, "w") as f:
            json.dump(evidence.to_dict(), f, indent=2)
        logger.info("Recorded evidence: %s -> %s", evidence.id, path)
        return path

    def list_all(self) -> List[AxisEvidence]:
        """Return every non-stale evidence record on disk."""
        if not self._dir.exists():
            return []
        cutoff = datetime.utcnow() - timedelta(days=MAX_AGE_DAYS)
        records: list[AxisEvidence] = []
        for p in sorted(self._dir.glob("*.json")):
            ev = self._read(p)
            if ev is None or self._is_stale(ev, cutoff):
                continue
            records.append(ev)
        return records

    # ---- Query ----------------------------------------------------------

    def query(
        self, axis_name: str, model_name: str,
        task_name: Optional[str] = None, method_name: Optional[str] = None,
    ) -> List[AxisEvidence]:
        """Find matching records.

        Exact model match first; then model-family match with decayed
        confidence.  Stale records (>MAX_AGE_DAYS) are excluded.
        """
        all_records = self.list_all()
        target_family = _model_family(model_name)
        exact: list[AxisEvidence] = []
        family: list[AxisEvidence] = []
        for ev in all_records:
            if ev.axis_name != axis_name:
                continue
            if task_name and ev.task_name != task_name:
                continue
            if method_name and ev.method_name != method_name:
                continue
            if ev.model_name == model_name:
                exact.append(ev)
            elif ev.model_family == target_family:
                family.append(ev)
        if exact:
            return exact
        decayed: list[AxisEvidence] = []
        for ev in family:
            decayed.append(AxisEvidence(
                axis_name=ev.axis_name, model_name=ev.model_name,
                task_name=ev.task_name, method_name=ev.method_name,
                tested_values=ev.tested_values, scores=ev.scores,
                dominant_values=ev.dominant_values, margin=ev.margin,
                confidence=ev.confidence * CROSS_MODEL_CONFIDENCE_DECAY,
                n_samples=ev.n_samples, created_at=ev.created_at,
                source=ev.source, notes=ev.notes,
            ))
        return decayed

    # ---- Reductions -----------------------------------------------------

    def get_reductions(
        self, model_name: str,
        task_name: Optional[str] = None, method_name: Optional[str] = None,
    ) -> Dict[str, AxisReduction]:
        """Compute per-axis reductions from all relevant evidence.

        Same axis + same task: intersection of dominant_values.
        Same axis + different tasks: union.
        Empty intersection falls back to full axis.
        """
        all_records = self.list_all()
        target_family = _model_family(model_name)
        by_axis: Dict[str, list[AxisEvidence]] = defaultdict(list)
        for ev in all_records:
            if method_name and ev.method_name != method_name:
                continue
            is_exact = ev.model_name == model_name
            is_family = ev.model_family == target_family
            if not is_exact and not is_family:
                continue
            by_axis[ev.axis_name].append(ev)
        reductions: Dict[str, AxisReduction] = {}
        for axis, records in by_axis.items():
            reduction = self._reduce_axis(axis, records, task_name)
            if reduction is not None:
                reductions[axis] = reduction
        return reductions

    def _reduce_axis(
        self, axis_name: str, records: List[AxisEvidence],
        filter_task: Optional[str],
    ) -> Optional[AxisReduction]:
        """Merge multiple evidence records for one axis."""
        if not records:
            return None
        by_task: Dict[str, list[AxisEvidence]] = defaultdict(list)
        for ev in records:
            by_task[ev.task_name].append(ev)
        per_task_dominant: list[set[str]] = []
        all_tested: set[str] = set()
        evidence_ids: list[str] = []
        total_confidence = 0.0
        n_records = 0
        for task, evs in by_task.items():
            dominant_set: Optional[set[str]] = None
            for ev in evs:
                evidence_ids.append(ev.id)
                total_confidence += ev.confidence
                n_records += 1
                all_tested.update(ev.tested_values)
                s = set(ev.dominant_values)
                dominant_set = s if dominant_set is None else (dominant_set & s)
            if dominant_set:
                per_task_dominant.append(dominant_set)
        if not per_task_dominant:
            return None
        if filter_task:
            keep = per_task_dominant[0]
            for s in per_task_dominant[1:]:
                keep = keep & s
        else:
            keep: set[str] = set()
            for s in per_task_dominant:
                keep = keep | s
        if not keep:
            logger.warning(
                "Evidence for axis %s yielded empty keep set; "
                "falling back to full axis", axis_name,
            )
            return None
        removed = sorted(all_tested - keep)
        avg_conf = total_confidence / n_records if n_records else 0.0
        return AxisReduction(
            axis_name=axis_name, keep_values=sorted(keep),
            removed_values=removed, evidence_ids=evidence_ids,
            confidence=avg_conf,
        )

    # ---- GCS sync -------------------------------------------------------

    def upload_to_gcs(self, bucket_name: str = "wisent-gcp-bucket") -> int:
        """Upload all local ledger files to GCS. Returns count uploaded."""
        from google.cloud import storage
        if not self._dir.exists():
            return 0
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        count = 0
        for p in sorted(self._dir.glob("*.json")):
            blob = bucket.blob(f"{_GCS_PREFIX}/{p.name}")
            blob.upload_from_filename(str(p))
            count += 1
        logger.info("Uploaded %d evidence files to gs://%s/%s/",
                     count, bucket_name, _GCS_PREFIX)
        return count

    def download_from_gcs(self, bucket_name: str = "wisent-gcp-bucket") -> int:
        """Download evidence from GCS, merge with local (keep newer)."""
        from google.cloud import storage
        self._dir.mkdir(parents=True, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=f"{_GCS_PREFIX}/"))
        count = 0
        for blob in blobs:
            fname = blob.name.split("/")[-1]
            if not fname.endswith(".json"):
                continue
            local_path = self._dir / fname
            if local_path.exists():
                local_ev = self._read(local_path)
                remote_data = json.loads(blob.download_as_text())
                remote_ev = AxisEvidence.from_dict(remote_data)
                if local_ev and local_ev.created_at >= remote_ev.created_at:
                    continue
            blob.download_to_filename(str(local_path))
            count += 1
        logger.info("Downloaded %d evidence files from GCS", count)
        return count

    # ---- Helpers --------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> Optional[AxisEvidence]:
        try:
            with open(path) as f:
                return AxisEvidence.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Skipping invalid evidence file %s: %s", path, exc)
            return None

    @staticmethod
    def _is_stale(ev: AxisEvidence, cutoff: datetime) -> bool:
        try:
            return datetime.fromisoformat(ev.created_at) < cutoff
        except (ValueError, TypeError):
            return False
