import sys
from typing import Dict, Callable, Optional, TypeVar, List, Generic

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(
            self,
            name: str,
            expose_module: Optional[str] = None,
            allow_override: bool = False
    ):
        self.name = name
        self._registry: Dict[str, T] = {}
        self._allow_override = allow_override
        self._expose_module: Optional[sys.modules] = None

        if expose_module:
            self._set_expose_module(expose_module)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    @property
    def list(self) -> List[str]:
        return list(set(self._registry.keys()))

    def get(self, name: str) -> T:
        if name not in self._registry:
            raise KeyError(f"{self.name} '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def _set_expose_module(self, module_name: str) -> None:
        if module_name not in sys.modules:
            raise ValueError(f"Module '{module_name}' not found. Cannot expose registry items.")
        self._expose_module = sys.modules[module_name]

        if not hasattr(self._expose_module, "__all__"):
            self._expose_module.__all__ = []

    def _expose_item(self, name: str, item: T) -> None:
        if self._expose_module is None:
            return

        setattr(self._expose_module, name, item)

        if name not in self._expose_module.__all__:
            self._expose_module.__all__.append(name)

    def _remove_expose_item(self, name: str) -> None:
        if self._expose_module is None:
            return

        if hasattr(self._expose_module, name):
            delattr(self._expose_module, name)

        if name in self._expose_module.__all__:
            self._expose_module.__all__.remove(name)

    def register(self, name: Optional[str] = None, override: Optional[bool] = None) -> Callable[[T], T]:
        def wrapper(item: T) -> T:
            key = name or getattr(item, "__name__", str(item))
            allow_override = override if override is not None else self._allow_override

            if not allow_override and key in self._registry:
                raise ValueError(f"{self.name} '{key}' already registered. Available: {list(self._registry.keys())}")

            self._registry[key] = item
            self._expose_item(key, item)

            return item
        return wrapper

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            raise KeyError(f"{self.name} '{name}' not found.")
        del self._registry[name]
        self._remove_expose_item(name)