"""
Base rotator class providing shared plugin discovery and resolution logic.

This module extracts common patterns from EvaluatorRotator, DataLoaderRotator,
and SteeringMethodRotator into a reusable base class.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

__all__ = [
    "BaseRotator",
    "RotatorError",
]

logger = logging.getLogger(__name__)


class RotatorError(RuntimeError):
    """Base error for rotator operations."""
    pass


# Type variable for the plugin base class
T = TypeVar("T")


class BaseRotator(ABC, Generic[T]):
    """
    Abstract base class for plugin rotators.

    Provides common functionality for discovering, registering, resolving,
    and using plugins (evaluators, data loaders, steering methods, etc.).

    Subclasses must implement:
        - _get_registry_class(): Returns the base class that holds the registry
        - _get_error_class(): Returns the error class to use for this rotator
        - _get_plugin_type_name(): Returns human-readable name (e.g., "evaluator")

    Type Parameters:
        T: The base plugin class type (e.g., BaseEvaluator, BaseDataLoader)

    Example:
        >>> class MyRotator(BaseRotator[MyPlugin]):
        ...     def _get_registry_class(self):
        ...         return MyPlugin
        ...     def _get_error_class(self):
        ...         return MyPluginError
        ...     def _get_plugin_type_name(self):
        ...         return "plugin"
    """

    def __init__(
        self,
        plugin: Union[str, T, Type[T], None] = None,
        location: Union[str, Path] = "",
        autoload: bool = True,
        scope_prefix: Optional[str] = None,
        **default_kwargs: Any,
    ) -> None:
        """
        Initialize the rotator.

        Args:
            plugin: Plugin name, instance, class, or None for auto-selection.
            location: Dotted module path or directory path for discovery.
            autoload: Whether to auto-discover plugins on init.
            scope_prefix: Optional prefix to filter registered plugins by module.
            **default_kwargs: Default kwargs to pass to plugin instantiation.
        """
        self._location = location
        self._scope_prefix = scope_prefix or (
            location if isinstance(location, str) else ""
        )
        self._default_kwargs = default_kwargs

        if autoload and location:
            self.discover(location)

        self._plugin = self._resolve(plugin, **default_kwargs)

    @abstractmethod
    def _get_registry_class(self) -> Type[T]:
        """Return the base class that holds the plugin registry."""
        raise NotImplementedError

    @abstractmethod
    def _get_error_class(self) -> Type[Exception]:
        """Return the error class to use for this rotator."""
        raise NotImplementedError

    @abstractmethod
    def _get_plugin_type_name(self) -> str:
        """Return human-readable plugin type name (e.g., 'evaluator')."""
        raise NotImplementedError

    def discover(self, location: Union[str, Path]) -> None:
        """
        Import all plugin modules so subclasses self-register.

        Args:
            location: Either a dotted module path (e.g., 'wisent.core.evaluators.oracles')
                     or an existing directory path.

        Raises:
            RotatorError: If location is invalid or cannot be imported.
        """
        loc_path = Path(str(location))

        # If it's an existing directory, import all .py files
        if loc_path.exists() and loc_path.is_dir():
            self._import_all_py_in_dir(loc_path)
            return

        # Otherwise treat as dotted module path
        if not isinstance(location, str):
            raise self._get_error_class()(
                f"Invalid {self._get_plugin_type_name()}s location: {location!r}. "
                "Provide a dotted module path or a directory."
            )

        try:
            pkg = importlib.import_module(location)
        except ModuleNotFoundError as exc:
            raise self._get_error_class()(
                f"Cannot import {self._get_plugin_type_name()} package {location!r}. "
                "Use dotted path (no leading slash) and ensure your project root is on PYTHONPATH."
            ) from exc

        # Get search paths (supports namespace packages)
        search_paths = list(getattr(pkg, "__path__", []))
        if not search_paths:
            pkg_file = getattr(pkg, "__file__", None)
            if pkg_file:
                search_paths = [str(Path(pkg_file).parent)]

        # Import all submodules
        for finder, name, ispkg in pkgutil.iter_modules(search_paths):
            if name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{location}.{name}")
            except Exception as e:
                logger.warning(f"Failed to import {location}.{name}: {e}")

    def _import_all_py_in_dir(self, directory: Path) -> None:
        """
        Import all .py files in a directory.

        Args:
            directory: Path to directory containing .py files.
        """
        plugin_type = self._get_plugin_type_name()
        for py in directory.glob("*.py"):
            if py.name.startswith("_"):
                continue
            mod_name = f"_dyn_{plugin_type}s_{py.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, py)
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.warning(f"Failed to load {py}: {e}")

    def _get_registry(self) -> Dict[str, Type[T]]:
        """Get the plugin registry from the base class."""
        registry_class = self._get_registry_class()
        if hasattr(registry_class, "list_registered"):
            return registry_class.list_registered()
        return {}

    def _get_scoped_registry(self) -> Dict[str, Type[T]]:
        """Get registry filtered by scope prefix."""
        reg = self._get_registry()
        if not self._scope_prefix:
            return reg
        return {
            name: cls for name, cls in reg.items()
            if cls.__module__.startswith(self._scope_prefix)
        }

    def list_plugins(self, scope_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered plugins.

        Args:
            scope_prefix: Optional module prefix to filter by.

        Returns:
            List of dicts with plugin info (name, description, class).
        """
        reg = self._get_registry()
        if scope_prefix:
            reg = {n: c for n, c in reg.items() if c.__module__.startswith(scope_prefix)}

        return [
            {
                "name": name,
                "description": getattr(cls, "description", ""),
                "class": f"{cls.__module__}.{cls.__name__}",
            }
            for name, cls in sorted(reg.items(), key=lambda kv: kv[0])
        ]

    def _resolve(
        self,
        plugin: Union[str, T, Type[T], None],
        **kwargs: Any,
    ) -> T:
        """
        Resolve a plugin from various input types.

        Args:
            plugin: Plugin name, instance, class, or None.
            **kwargs: Kwargs to pass to plugin constructor.

        Returns:
            Resolved plugin instance.

        Raises:
            RotatorError: If plugin cannot be resolved.
        """
        registry_class = self._get_registry_class()
        error_class = self._get_error_class()
        plugin_type = self._get_plugin_type_name()
        reg = self._get_scoped_registry()

        # None -> auto-select first available
        if plugin is None:
            if not reg:
                raise error_class(
                    f"No {plugin_type}s registered"
                    + (f" under {self._scope_prefix!r}" if self._scope_prefix else "")
                    + "."
                )
            first_cls = next(iter(sorted(reg.items(), key=lambda kv: kv[0])))[1]
            return first_cls(**kwargs)

        # Already an instance
        if isinstance(plugin, registry_class):
            # Merge kwargs
            if hasattr(plugin, "kwargs"):
                plugin.kwargs = {**kwargs, **plugin.kwargs}
            return plugin

        # A class (subclass of registry class)
        if inspect.isclass(plugin) and issubclass(plugin, registry_class):
            return plugin(**kwargs)

        # A string name
        if isinstance(plugin, str):
            if plugin not in reg:
                # Try full registry if not in scoped
                full_reg = self._get_registry()
                if plugin in full_reg:
                    return full_reg[plugin](**kwargs)
                raise error_class(
                    f"Unknown {plugin_type} {plugin!r}"
                    + (f" in scope {self._scope_prefix!r}" if self._scope_prefix else "")
                    + f". Available: {list(reg.keys())}"
                )
            return reg[plugin](**kwargs)

        raise TypeError(
            f"{plugin_type} must be None, a name (str), "
            f"{registry_class.__name__} instance, or {registry_class.__name__} subclass."
        )

    def use(self, plugin: Union[str, T, Type[T]], **kwargs: Any) -> None:
        """
        Switch to a different plugin.

        Args:
            plugin: Plugin name, instance, or class.
            **kwargs: Kwargs to pass to plugin constructor.
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}
        self._plugin = self._resolve(plugin, **merged_kwargs)

    @property
    def current(self) -> T:
        """Get the currently selected plugin instance."""
        return self._plugin
