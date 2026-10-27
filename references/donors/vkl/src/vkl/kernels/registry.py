from __future__ import annotations

from collections.abc import Callable

REGISTRY: dict[str, Callable[..., dict]] = {}


def register(name: str):
    def deco(fn: Callable[..., dict]):
        if name in REGISTRY:
            raise KeyError(f"Kernel already registered: {name}")
        REGISTRY[name] = fn
        return fn
    return deco
