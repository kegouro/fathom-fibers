from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .zeiss import inspect_image


def _inspect(paths: list[str], compute_hash: bool) -> int:
    for raw in paths:
        try:
            print(json.dumps(inspect_image(raw, compute_hash=compute_hash), indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"ERROR {raw}: {exc}", file=sys.stderr)
            return 1
    return 0


def _inventory(directory: str, output: str) -> int:
    root = Path(directory)
    paths = sorted([*root.glob("*.tif"), *root.glob("*.tiff"), *root.glob("*.TIF"), *root.glob("*.TIFF")])
    rows = []
    for path in paths:
        info = inspect_image(path, compute_hash=False)
        metadata = info.get("metadata", {})
        calibration = info.get("calibration") or {}
        rows.append({
            "filename": path.name,
            "format_id": info.get("format_id"),
            "width_px": info.get("width_px"),
            "height_px": info.get("height_px"),
            "pixel_size_nm": calibration.get("pixel_size_x_m", 0) * 1e9 if calibration else "",
            "magnification": metadata.get("ap_mag", ""),
            "field_width": metadata.get("ap_width", ""),
            "field_width_unit": metadata.get("ap_width__unit", ""),
            "field_height": metadata.get("ap_height", ""),
            "field_height_unit": metadata.get("ap_height__unit", ""),
            "eht_kv": metadata.get("ap_actualkv", ""),
            "wd_mm": metadata.get("ap_wd", ""),
            "detector": metadata.get("dp_detector_channel", ""),
            "date": metadata.get("ap_date", ""),
            "time": metadata.get("ap_time", ""),
            "width_crosscheck_relative": info.get("width_crosscheck_relative", ""),
            "height_crosscheck_relative": info.get("height_crosscheck_relative", ""),
        })
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="fathom-fibers", description="Fathom Fibers Quick MVP")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Launch the interactive application")
    gui.add_argument("path", nargs="?")

    inspect = sub.add_parser("inspect", help="Inspect Zeiss/TIFF metadata")
    inspect.add_argument("paths", nargs="+")
    inspect.add_argument("--hash", action="store_true")

    plugins = sub.add_parser("plugins", help="List discovered classical and model providers")
    plugins.add_argument("--load", action="store_true", help="Import providers to validate them")

    inventory = sub.add_parser("inventory", help="Create a CSV inventory of TIFF files")
    inventory.add_argument("directory")
    inventory.add_argument("--output", "-o", default="zeiss_inventory.csv")

    args = parser.parse_args()
    if args.command in {None, "gui"}:
        from .app import FiberQuickApp
        app = FiberQuickApp(getattr(args, "path", None))
        app.mainloop()
        return
    if args.command == "inspect":
        raise SystemExit(_inspect(args.paths, args.hash))
    if args.command == "plugins":
        from .plugin_registry import discover_classical, discover_models
        payload = {
            "classical": [provider.__dict__ if hasattr(provider, "__dict__") else {
                "name": provider.name, "group": provider.group, "value": provider.value,
                "distribution": provider.distribution, "error": provider.error
            } for provider in discover_classical(load=args.load)],
            "models": [provider.__dict__ if hasattr(provider, "__dict__") else {
                "name": provider.name, "group": provider.group, "value": provider.value,
                "distribution": provider.distribution, "error": provider.error
            } for provider in discover_models(load=args.load)],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "inventory":
        raise SystemExit(_inventory(args.directory, args.output))


if __name__ == "__main__":
    main()
