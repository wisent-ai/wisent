"""Constant profiles for per-model optimized constant values."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from wisent.core import constants as _constants_module

logger = logging.getLogger(__name__)

_PROFILES_DIR_NAME = "constant_profiles"
_GCS_PREFIX = "constant_profiles"


@dataclass
class ConstantProfile:
    """A set of optimized constant values for a specific model/task."""

    model_name: str
    task_name: Optional[str]
    constants: Dict[str, float]
    source: str
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> ConstantProfile:
        """Deserialize from dict."""
        return cls(
            model_name=data["model_name"],
            task_name=data.get("task_name"),
            constants=data["constants"],
            source=data["source"],
            metrics=data.get("metrics", {}),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


class ConstantProfileManager:
    """Manage saving, loading, activating, and syncing constant profiles."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        if profiles_dir is None:
            self._profiles_dir = Path.home() / ".wisent" / _PROFILES_DIR_NAME
        else:
            self._profiles_dir = profiles_dir
        self._defaults: Dict[str, float] = {}
        self._active_profile: Optional[ConstantProfile] = None

    @property
    def profiles_dir(self) -> Path:
        """Directory where profiles are stored."""
        return self._profiles_dir

    def _profile_path(self, model_name: str, task_name: Optional[str]) -> Path:
        """Get the file path for a profile."""
        safe_model = model_name.replace("/", "_").replace("\\", "_")
        if task_name:
            safe_task = task_name.replace("/", "_").replace("\\", "_")
            filename = f"{safe_model}_{safe_task}.json"
        else:
            filename = f"{safe_model}.json"
        return self._profiles_dir / filename

    def save(self, profile: ConstantProfile) -> Path:
        """Save a profile to disk."""
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self._profile_path(profile.model_name, profile.task_name)
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=_constants_module.JSON_INDENT)
        logger.info("Saved constant profile to %s", path)
        return path

    def load(self, model_name: str, task_name: Optional[str] = None) -> Optional[ConstantProfile]:
        """Load a profile. Tries task-specific first, then model-wide."""
        if task_name:
            path = self._profile_path(model_name, task_name)
            if path.exists():
                return self._read_profile(path)
        model_path = self._profile_path(model_name, None)
        if model_path.exists():
            return self._read_profile(model_path)
        return None

    def _read_profile(self, path: Path) -> ConstantProfile:
        """Read a profile from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return ConstantProfile.from_dict(data)

    def activate(self, profile: ConstantProfile) -> None:
        """Activate a profile by patching constants module via setattr."""
        from .registry import get_registry
        registry = get_registry()
        self._defaults = {}
        applied = 0
        for name, value in profile.constants.items():
            if name not in registry:
                logger.warning("Profile constant %s not in registry, skipping", name)
                continue
            meta = registry[name]
            old_val = getattr(_constants_module, name, None)
            if old_val is not None:
                self._defaults[name] = float(old_val)
            clamped = meta.clamp(value)
            cast = meta.cast_value(clamped)
            setattr(_constants_module, name, cast)
            applied += 1
        self._active_profile = profile
        logger.info("Activated profile: %d constants patched", applied)

    def deactivate(self) -> None:
        """Reset all patched constants to their defaults."""
        for name, default_val in self._defaults.items():
            setattr(_constants_module, name, default_val)
        logger.info("Deactivated profile: %d constants restored", len(self._defaults))
        self._defaults = {}
        self._active_profile = None

    @property
    def active_profile(self) -> Optional[ConstantProfile]:
        """Currently active profile, if any."""
        return self._active_profile

    def list_profiles(self) -> List[Path]:
        """List all saved profile files."""
        if not self._profiles_dir.exists():
            return []
        return sorted(self._profiles_dir.glob("*.json"))

    def delete(self, model_name: str, task_name: Optional[str] = None) -> bool:
        """Delete a saved profile."""
        path = self._profile_path(model_name, task_name)
        if path.exists():
            path.unlink()
            logger.info("Deleted profile: %s", path)
            return True
        return False

    def upload_to_gcs(self, profile: ConstantProfile, bucket_name: str = "wisent-gcp-bucket") -> str:
        """Upload a profile to GCS."""
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        safe_model = profile.model_name.replace("/", "_")
        if profile.task_name:
            safe_task = profile.task_name.replace("/", "_")
            blob_name = f"{_GCS_PREFIX}/{safe_model}_{safe_task}.json"
        else:
            blob_name = f"{_GCS_PREFIX}/{safe_model}.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(profile.to_dict(), indent=_constants_module.JSON_INDENT),
            content_type="application/json",
        )
        gcs_path = f"gs://{bucket_name}/{blob_name}"
        logger.info("Uploaded profile to %s", gcs_path)
        return gcs_path

    def download_from_gcs(
        self, model_name: str, task_name: Optional[str] = None,
        bucket_name: str = "wisent-gcp-bucket",
    ) -> Optional[ConstantProfile]:
        """Download a profile from GCS."""
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        safe_model = model_name.replace("/", "_")
        if task_name:
            safe_task = task_name.replace("/", "_")
            blob_name = f"{_GCS_PREFIX}/{safe_model}_{safe_task}.json"
        else:
            blob_name = f"{_GCS_PREFIX}/{safe_model}.json"
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.info("No GCS profile found at %s", blob_name)
            return None
        data = json.loads(blob.download_as_text())
        profile = ConstantProfile.from_dict(data)
        self.save(profile)
        logger.info("Downloaded profile from gs://%s/%s", bucket_name, blob_name)
        return profile
