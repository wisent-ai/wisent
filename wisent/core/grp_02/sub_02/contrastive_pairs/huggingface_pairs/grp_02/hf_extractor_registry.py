from __future__ import annotations

from typing import Type, Union
import importlib
import logging

from wisent.core.contrastive_pairs.huggingface_pairs.atoms import (
    HuggingFaceBenchmarkExtractor,
    UnsupportedHuggingFaceBenchmarkError,
)

from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import EXTRACTORS as _MANIFEST
from wisent.core.errors import InvalidValueError, InvalidDataFormatError

__all__ = [
    "register_extractor",
    "get_extractor",
]

LOG = logging.getLogger(__name__)

_REGISTRY: dict[str, Union[str, Type[HuggingFaceBenchmarkExtractor]]] = dict(_MANIFEST)


def register_extractor(name: str, ref: Union[str, Type[HuggingFaceBenchmarkExtractor]]) -> None:
    """
    Register a new HuggingFace extractor by name.

    arguments:
        name:
            Name/key for the extractor (case-insensitive).
        ref:
            Either a string "module_path:ClassName[.Inner]" or a subclass of
            HuggingFaceBenchmarkExtractor.

    raises:
        ValueError:
            If the name is empty or the string ref is malformed.
        TypeError:
            If the ref class does not subclass HuggingFaceBenchmarkExtractor.

    example:
        >>> from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import register_extractor
        >>> from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
        >>> class MyExtractor(HuggingFaceBenchmarkExtractor): ...
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

    if not issubclass(ref, HuggingFaceBenchmarkExtractor):
        raise TypeError(f"{getattr(ref, '__name__', ref)!r} must subclass HuggingFaceBenchmarkExtractor")

    _REGISTRY[key] = ref


def get_extractor(task_name: str) -> HuggingFaceBenchmarkExtractor:
    """
    Retrieve a registered HuggingFace extractor by task name.

    arguments:
        task_name:
            Name of the HuggingFace benchmark/task (e.g., "humaneval", "mbpp").
            Case-insensitive.

    returns:
        An instance of the corresponding HuggingFaceBenchmarkExtractor subclass.

    raises:
        UnsupportedHuggingFaceBenchmarkError:
            If no extractor is registered for the given task name.
        ImportError:
            If the extractor class cannot be imported/resolved.
        TypeError:
            If the resolved class does not subclass HuggingFaceBenchmarkExtractor.
    """

    key = (task_name or "").strip().lower()
    if not key:
        raise UnsupportedHuggingFaceBenchmarkError("Empty task name is not supported.")

    # Try exact match
    ref = _REGISTRY.get(key)
    if ref:
        return _instantiate(ref)

    raise UnsupportedHuggingFaceBenchmarkError(
        f"No extractor registered for HuggingFace task '{task_name}'. "
        f"Known: {', '.join(sorted(_REGISTRY)) or '(none)'}"
    )

def _instantiate(ref: Union[str, Type[HuggingFaceBenchmarkExtractor]]) -> HuggingFaceBenchmarkExtractor:
    """
    Instantiate an extractor from a string reference or class.

    arguments:
        ref:
            Either a string "module_path:ClassName[.Inner]" or a subclass of
            HuggingFaceBenchmarkExtractor.

    returns:
        An instance of the corresponding HuggingFaceBenchmarkExtractor subclass.

    raises:
        ImportError:
            If the extractor class cannot be imported/resolved.
        TypeError:
            If the resolved class does not subclass HuggingFaceBenchmarkExtractor.
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

    if not isinstance(obj, type) or not issubclass(obj, HuggingFaceBenchmarkExtractor):
        raise TypeError(f"Resolved object '{obj}' is not a HuggingFaceBenchmarkExtractor subclass.")
    return obj()
