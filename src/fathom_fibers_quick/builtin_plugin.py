from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BuiltinPlugin:
    plugin_id: str = "fathom_fibers_quick.builtin"
    api_version: str = "1"
    capabilities: tuple[str, ...] = (
        "zeiss_cz_sem_reader",
        "manual_caliper",
        "assisted_edge_snap",
        "assisted_local_one_click",
        "fiber_size_clustering",
    )


PLUGIN = BuiltinPlugin()
