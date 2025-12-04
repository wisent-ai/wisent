"""
Data loader rotator for discovering and selecting data loaders.

Uses BaseRotator for common plugin discovery and resolution logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from wisent.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.utils.base_rotator import BaseRotator

__all__ = [
    "DataLoaderRotator",
]


class DataLoaderRotator(BaseRotator[BaseDataLoader]):
    """
    Discover/select a data loader and use it to load data.

    Extends BaseRotator with data loader-specific functionality:
    - Scoped registry filtering by module prefix
    - Load method with kwargs merging
    """

    def __init__(
        self,
        loader: Union[str, BaseDataLoader, Type[BaseDataLoader], None] = None,
        loaders_location: Union[str, Path] = "wisent.core.data_loaders.loaders",
        autoload: bool = True,
        **default_loader_kwargs: Any,
    ) -> None:
        """
        Initialize the data loader rotator.

        Args:
            loader: Loader name, instance, class, or None for auto-selection.
            loaders_location: Module path or directory for loader discovery.
            autoload: Whether to auto-discover loaders on init.
            **default_loader_kwargs: Default kwargs passed to loader.
        """
        scope_prefix = (
            loaders_location if isinstance(loaders_location, str)
            else Path(loaders_location).as_posix().replace("/", ".")
        )

        super().__init__(
            plugin=loader,
            location=loaders_location,
            autoload=autoload,
            scope_prefix=scope_prefix,
            **default_loader_kwargs,
        )

    def _get_registry_class(self) -> Type[BaseDataLoader]:
        return BaseDataLoader

    def _get_error_class(self) -> Type[Exception]:
        return DataLoaderError

    def _get_plugin_type_name(self) -> str:
        return "loader"

    # Keep static method for backward compatibility
    @staticmethod
    def discover_loaders(location: Union[str, Path]) -> None:
        """
        Import all loader modules so BaseDataLoader subclasses self-register.

        Static method for backward compatibility.
        """
        rotator = DataLoaderRotator.__new__(DataLoaderRotator)
        rotator._scope_prefix = ""
        rotator.discover(location)

    @staticmethod
    def list_loaders(scope_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered loaders.

        Args:
            scope_prefix: Optional module prefix to filter by.

        Returns:
            List of dicts with loader info.
        """
        reg = BaseDataLoader.list_registered()
        if scope_prefix:
            reg = {n: c for n, c in reg.items() if c.__module__.startswith(scope_prefix)}
        return [
            {
                "name": n,
                "description": getattr(c, "description", ""),
                "class": f"{c.__module__}.{c.__name__}",
            }
            for n, c in sorted(reg.items(), key=lambda kv: kv[0])
        ]

    def use(self, loader: Union[str, BaseDataLoader, Type[BaseDataLoader]], **kwargs: Any) -> None:
        """Switch to a different loader."""
        merged_kwargs = {**self._default_kwargs, **kwargs}
        self._plugin = self._resolve(loader, **merged_kwargs)

    def load(self, **kwargs: Any) -> LoadDataResult:
        """
        Load data using the current loader.

        Args:
            **kwargs: Kwargs passed to loader's load method.

        Returns:
            LoadDataResult with loaded data.
        """
        merged = {**getattr(self._plugin, "kwargs", {}), **kwargs}
        return self._plugin.load(**merged)
