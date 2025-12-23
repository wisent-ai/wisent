from __future__ import annotations

from typing import Type, Union
import importlib
import logging

from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import (
    LMEvalBenchmarkExtractor,
    UnsupportedLMEvalBenchmarkError,
)
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_manifest import EXTRACTORS as _LM_MANIFEST
from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import HF_EXTRACTORS as _HF_MANIFEST
from wisent.core.errors import InvalidValueError, InvalidDataFormatError

__all__ = [
    "register_extractor",
    "get_extractor",
]

LOG = logging.getLogger(__name__)

# Combine LM-eval and HuggingFace manifests (HF takes precedence for overlapping keys)
_COMBINED_MANIFEST = {**_LM_MANIFEST, **_HF_MANIFEST}
_REGISTRY: dict[str, Union[str, Type[LMEvalBenchmarkExtractor]]] = {k.lower(): v for k, v in _COMBINED_MANIFEST.items()}


def register_extractor(name: str, ref: Union[str, Type[LMEvalBenchmarkExtractor]]) -> None:
    """
    Register a new extractor by name.
    arguments:
        name:
            Name/key for the extractor (case-insensitive).
        ref:
            Either a string "module_path:ClassName[.Inner]" or a subclass of
            LMEvalBenchmarkExtractor.
    raises:
        ValueError:
            If the name is empty or the string ref is malformed.
        TypeError:
            If the ref class does not subclass LMEvalBenchmarkExtractor.

    example:
        >>> from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import register_extractor
        >>> from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
        >>> class MyExtractor(LMEvalBenchmarkExtractor): ...
        >>> register_extractor("mytask", MyExtractor)
        >>> register_extractor("mytask2", "my_module:MyExtractor")
    """
    key = (name or "").strip().lower()
    if not key:
        raise InvalidValueError(param_name="name/key", actual="empty string", expected="non-empty string")

    if isinstance(ref, str):
        if ":" not in ref:
            raise InvalidDataFormatError(reason="String ref must be 'module_path:ClassName[.Inner]'.")
        _REGISTRY[key] = ref
        return

    if not issubclass(ref, LMEvalBenchmarkExtractor):
        raise TypeError(f"{getattr(ref, '__name__', ref)!r} must subclass LMEvalBenchmarkExtractor")
    
    _REGISTRY[key] = ref


def get_extractor(task_name: str) -> LMEvalBenchmarkExtractor:
    """
    Retrieve a registered extractor by task name.

    arguments:
        task_name:
            Name of the lm-eval benchmark/task (e.g., "winogrande").
            Case-insensitive. Tries exact match first, then prefix match.
            For example, "mmlu_anatomy" will match "mmlu" extractor.

    returns:
        An instance of the corresponding LMEvalBenchmarkExtractor subclass.

    raises:
        UnsupportedLMEvalBenchmarkError:
            If no extractor is registered for the given task name.
        ImportError:
            If the extractor class cannot be imported/resolved.
        TypeError:
            If the resolved class does not subclass LMEvalBenchmarkExtractor.
    """

    key = (task_name or "").strip().lower()
    if not key:
        raise UnsupportedLMEvalBenchmarkError("Empty task name is not supported.")

    # Try exact match first
    ref = _REGISTRY.get(key)
    if ref:
        return _instantiate(ref)

    # Try prefix matching for hierarchical task names
    # This handles cases like AraDiCE_ArabicMMLU_high_humanities_history_lev -> aradice
    # Sort prefixes by length descending to match longest prefix first
    PREFIX_FALLBACKS = {
        "aradice_": "aradice",
        "aexams_": "aexams",
        "afrimgsm_": "afrimgsm",
        "afrimmlu_": "afrimmlu",
        "afrobench_": "afrobench",
        "afridiacritics_": "afrobench",
        "mmlu_": "mmlu",
        "bigbench_": "bigbench",
    }
    for prefix, fallback_key in PREFIX_FALLBACKS.items():
        if key.startswith(prefix) and fallback_key in _REGISTRY:
            LOG.info(f"Using prefix fallback: '{task_name}' -> '{fallback_key}'")
            return _instantiate(_REGISTRY[fallback_key])

    raise UnsupportedLMEvalBenchmarkError(
        f"No extractor registered for task '{task_name}'. "
        f"Known: {', '.join(sorted(_REGISTRY)) or '(none)'}"
    )

def _instantiate(ref: Union[str, Type[LMEvalBenchmarkExtractor]]) -> Union[LMEvalBenchmarkExtractor, HuggingFaceBenchmarkExtractor]:
    """
    Instantiate an extractor from a string reference or class.

    arguments:
        ref:
            Either a string "module_path:ClassName[.Inner]" or a subclass of
            LMEvalBenchmarkExtractor or HuggingFaceBenchmarkExtractor.

    returns:
        An instance of the corresponding extractor subclass.

    raises:
        ImportError:
            If the extractor class cannot be imported/resolved.
        TypeError:
            If the resolved class does not subclass LMEvalBenchmarkExtractor or HuggingFaceBenchmarkExtractor.
    """
    if not isinstance(ref, str):
        return ref()

    module_path, attr_path = ref.split(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(f"Cannot import module '{module_path}' for extractor '{ref}'.") from exc

    obj = mod
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ImportError(f"Extractor class '{attr_path}' not found in '{module_path}'.") from exc

    # Accept both LMEval and HuggingFace extractors
    if not isinstance(obj, type) or not (issubclass(obj, LMEvalBenchmarkExtractor) or issubclass(obj, HuggingFaceBenchmarkExtractor)):
        raise TypeError(f"Resolved object '{obj}' is not a LMEvalBenchmarkExtractor or HuggingFaceBenchmarkExtractor subclass.")
    return obj()
