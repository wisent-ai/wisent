"""Persist optimization studies to HF for resumable runs."""
from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
    HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name,
)


def _study_hf_dir(model: str, benchmark: str, method: str) -> str:
    safe = model_to_safe_name(model)
    return f"studies/{safe}/{benchmark}/{method.lower()}"


def download_optuna_db(
    model: str, benchmark: str, method: str, local_dir: str,
) -> Optional[str]:
    """Download Optuna SQLite DB from HF. Returns local path or None."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _get_hf_token,
    )
    from huggingface_hub import hf_hub_download
    hf_path = f"{_study_hf_dir(model, benchmark, method)}/study.db"
    local_path = os.path.join(local_dir, "study.db")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hf_path,
            repo_type=HF_REPO_TYPE, token=_get_hf_token(),
            local_dir=local_dir,
        )
        if os.path.exists(downloaded):
            if downloaded != local_path:
                shutil.copy2(downloaded, local_path)
            print(f"  [study] Loaded existing study from HF: {hf_path}")
            return local_path
    except Exception:
        pass
    return None


def upload_optuna_db(
    model: str, benchmark: str, method: str, local_db_path: str,
) -> None:
    """Upload Optuna SQLite DB to HF."""
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            _get_api,
        )
        hf_path = f"{_study_hf_dir(model, benchmark, method)}/study.db"
        api = _get_api()
        api.upload_file(
            path_or_fileobj=local_db_path, path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
        print(f"  [study] Uploaded study to HF: {hf_path}")
    except Exception as exc:
        print(f"  [study] Upload failed: {exc}")


def download_hyperopt_trials(
    model: str, benchmark: str, method: str,
) -> Optional[object]:
    """Download Hyperopt Trials pickle from HF. Returns Trials or None."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _get_hf_token,
    )
    from huggingface_hub import hf_hub_download
    hf_path = f"{_study_hf_dir(model, benchmark, method)}/trials.pkl"
    try:
        with tempfile.TemporaryDirectory() as td:
            local = hf_hub_download(
                repo_id=HF_REPO_ID, filename=hf_path,
                repo_type=HF_REPO_TYPE, token=_get_hf_token(),
                local_dir=td,
            )
            with open(local, "rb") as f:
                trials = pickle.load(f)
            print(f"  [study] Loaded existing trials from HF: {hf_path}")
            return trials
    except Exception:
        return None


def upload_hyperopt_trials(
    model: str, benchmark: str, method: str, trials: object,
) -> None:
    """Upload Hyperopt Trials pickle to HF."""
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            _get_api,
        )
        hf_path = f"{_study_hf_dir(model, benchmark, method)}/trials.pkl"
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(trials, tmp)
            tmp_path = tmp.name
        try:
            api = _get_api()
            api.upload_file(
                path_or_fileobj=tmp_path, path_in_repo=hf_path,
                repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
            )
            print(f"  [study] Uploaded trials to HF: {hf_path}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"  [study] Upload failed: {exc}")
