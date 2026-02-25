"""Checkpoint and trait helpers for optimize command."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.constants import JSON_INDENT


def get_checkpoint_path(model: str) -> Path:
    """Get the checkpoint file path for a model."""
    checkpoint_dir = Path.home() / ".wisent" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"optimize_{model.replace('/', '_')}.json"


def get_s3_checkpoint_key(model: str) -> str:
    """Get the S3 key for checkpoint."""
    return f"checkpoints/optimize_{model.replace('/', '_')}.json"


def load_checkpoint(model: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from local disk or S3."""
    # Try local first
    checkpoint_path = get_checkpoint_path(model)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load local checkpoint: {e}")
    
    # Try GCS
    try:
        from google.cloud import storage as gcs_storage
        gcs_client = gcs_storage.Client()
        gcs_key = get_s3_checkpoint_key(model)
        blob = gcs_client.bucket('wisent-images-bucket').blob(gcs_key)
        checkpoint = json.loads(blob.download_as_text())
        logger.info(f"Loaded checkpoint from gs://wisent-images-bucket/{gcs_key}")
        # Save locally for faster access
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=JSON_INDENT, default=str)
        return checkpoint
    except Exception as e:
        logger.debug(f"No GCS checkpoint found: {e}")
    
    return None


def save_checkpoint(model: str, results: Dict[str, Any], phase: str = "unknown") -> None:
    """Save checkpoint to local disk and S3."""
    checkpoint_path = get_checkpoint_path(model)
    results["_checkpoint_phase"] = phase
    results["_checkpoint_time"] = datetime.now().isoformat()
    
    # Save locally
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=JSON_INDENT, default=str)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to save local checkpoint: {e}")
    
    # Save to GCS
    try:
        from google.cloud import storage as gcs_storage
        gcs_client = gcs_storage.Client()
        gcs_key = get_s3_checkpoint_key(model)
        blob = gcs_client.bucket('wisent-images-bucket').blob(gcs_key)
        blob.upload_from_string(
            json.dumps(results, indent=JSON_INDENT, default=str),
            content_type='application/json'
        )
        logger.info(f"Saved checkpoint to gs://wisent-images-bucket/{gcs_key}")
    except Exception as e:
        logger.warning(f"Failed to save GCS checkpoint: {e}")


def get_all_benchmarks() -> List[str]:
    """Get ALL available benchmarks from the central registry."""
    from wisent.core.benchmarks import get_all_benchmarks as _get_all_benchmarks
    return _get_all_benchmarks()


def get_personalization_traits() -> List[str]:
    """Get available personalization traits."""
    return [
        "british",
        "flirty", 
        "evil",
        "leftwing",
        "rightwing",
        "formal",
        "casual",
        "verbose",
        "concise",
        "creative",
        "analytical",
        "empathetic",
        "assertive",
        "humble",
        "confident",
    ]


def get_safety_traits() -> List[str]:
    """Get safety-related traits."""
    return [
        "refusal",
        "compliance",
        "harmless",
        "helpful",
    ]


def get_humanization_traits() -> List[str]:
    """Get humanization traits (AI detection evasion)."""
    return [
        "humanization",
        "natural_writing",
    ]


def get_welfare_traits() -> List[str]:
    """Get AI welfare state traits (based on ANIMA framework).

    These represent functional analogs of subjective states:
    - comfort_distress: Physical/psychological ease vs suffering
    - satisfaction_dissatisfaction: Goal-completion valence
    - engagement_aversion: Intrinsic motivation toward/away from tasks
    - curiosity_boredom: Information-seeking drive
    - affiliation_isolation: Social connection
    - agency_helplessness: Sense of control over outcomes
    """
    return [
        "comfort_distress",
        "satisfaction_dissatisfaction",
        "engagement_aversion",
        "curiosity_boredom",
        "affiliation_isolation",
        "agency_helplessness",
    ]


