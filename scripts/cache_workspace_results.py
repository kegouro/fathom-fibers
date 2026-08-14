#!/usr/bin/env python3
"""Populate the workspace full-result cache for a dataset, headless.

Reuses the exact staged orchestration of the Qt workspace controller so the
desktop application loads full per-image samples, overlays and distributions
without rerunning algorithms.

Usage:
    python scripts/cache_workspace_results.py --dataset <dir> [--case ZEISS_003]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.workspace import (
    WorkspaceCache,
    compute_comparison_staged,
    load_workspace_dataset,
    resolve_matlab_cache_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory (16 TIFFs)")
    parser.add_argument("--case", help="Restrict to one case id or image stem")
    parser.add_argument("--repo", default=".", help="Repository root for the cache")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    dataset = load_workspace_dataset(args.dataset, repo=repo)
    matlab_root = resolve_matlab_cache_root(dataset.source_dir, repo=repo)
    engine = FathomEngine()
    cache = WorkspaceCache(repo)
    selected = dataset.images
    if args.case:
        selected = [
            image
            for image in dataset.images
            if image.case_id == args.case or image.stem == args.case
        ]
        if not selected:
            print(f"No image matches {args.case!r}", file=sys.stderr)
            return 2
    started = time.monotonic()
    for image in selected:
        if cache.has_full(image.stem):
            print(f"cached  {image.filename}")
            continue
        step_started = time.monotonic()
        scientific = engine.open_image(image.absolute_path)
        comparison = compute_comparison_staged(
            engine,
            scientific,
            matlab_cache_root=matlab_root,
            progress=lambda message: print(f"  {message}", flush=True),
        )
        cache.store_comparison(image.stem, comparison)
        print(
            f"stored  {image.filename} ({time.monotonic() - step_started:.0f}s, "
            f"MATLAB cache: {matlab_root is not None})"
        )
    print(f"done in {time.monotonic() - started:.0f}s; MATLAB cache root: {matlab_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
