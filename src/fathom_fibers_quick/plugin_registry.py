from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .plugins import ENTRY_POINT_GROUP_CLASSICAL, ENTRY_POINT_GROUP_MODELS


@dataclass(frozen=True, slots=True)
class DiscoveredProvider:
    name: str
    group: str
    value: str
    distribution: str | None
    object: Any | None = None
    error: str | None = None


def discover(group: str, *, load: bool = False) -> list[DiscoveredProvider]:
    """Discover providers without importing heavy model packages unless requested."""
    providers: list[DiscoveredProvider] = []
    entry_points = metadata.entry_points().select(group=group)
    for entry_point in sorted(entry_points, key=lambda item: item.name):
        distribution = entry_point.dist.name if entry_point.dist else None
        loaded = None
        error = None
        if load:
            try:
                loaded = entry_point.load()
            except Exception as exc:  # plugin failures must remain isolated
                error = f"{type(exc).__name__}: {exc}"
        providers.append(
            DiscoveredProvider(
                name=entry_point.name,
                group=group,
                value=entry_point.value,
                distribution=distribution,
                object=loaded,
                error=error,
            )
        )
    return providers


def discover_classical(*, load: bool = False) -> list[DiscoveredProvider]:
    providers = discover(ENTRY_POINT_GROUP_CLASSICAL, load=load)
    if not any(provider.name == "builtin" for provider in providers):
        loaded = None
        error = None
        if load:
            try:
                from .builtin_plugin import PLUGIN
                loaded = PLUGIN
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        providers.insert(0, DiscoveredProvider(
            name="builtin",
            group=ENTRY_POINT_GROUP_CLASSICAL,
            value="fathom_fibers_quick.builtin_plugin:PLUGIN",
            distribution="source-tree",
            object=loaded,
            error=error,
        ))
    return providers


def discover_models(*, load: bool = False) -> list[DiscoveredProvider]:
    return discover(ENTRY_POINT_GROUP_MODELS, load=load)
