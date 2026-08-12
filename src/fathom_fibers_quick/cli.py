from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from .zeiss import inspect_image


def _inspect(paths: list[str], compute_hash: bool) -> int:
    for raw in paths:
        try:
            print(
                json.dumps(
                    inspect_image(raw, compute_hash=compute_hash), indent=2, ensure_ascii=False
                )
            )
        except Exception as exc:
            print(f"ERROR {raw}: {exc}", file=sys.stderr)
            return 1
    return 0


def _inventory(directory: str, output: str) -> int:
    root = Path(directory)
    paths = sorted(
        [*root.glob("*.tif"), *root.glob("*.tiff"), *root.glob("*.TIF"), *root.glob("*.TIFF")]
    )
    rows = []
    for path in paths:
        info = inspect_image(path, compute_hash=False)
        metadata = info.get("metadata", {})
        calibration = info.get("calibration") or {}
        rows.append(
            {
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
            }
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fathom-fibers",
        description="Scientific fiber measurement engine and desktop workspace",
    )
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Launch the interactive application")
    gui.add_argument("path", nargs="?")
    gui.add_argument(
        "--smoke-test", action="store_true", help="Run an offscreen-safe Qt smoke test"
    )

    inspect = sub.add_parser("inspect", help="Inspect Zeiss/TIFF metadata")
    inspect.add_argument("paths", nargs="+")
    inspect.add_argument("--hash", action="store_true")

    plugins = sub.add_parser("plugins", help="List discovered classical and model providers")
    plugins.add_argument("--load", action="store_true", help="Import providers to validate them")

    inventory = sub.add_parser("inventory", help="Create a CSV inventory of TIFF files")
    inventory.add_argument("directory")
    inventory.add_argument("--output", "-o", default="zeiss_inventory.csv")

    eval_roi = sub.add_parser("evaluate-auto-roi", help="Evaluate Auto-ROI campaign on Zeiss TIFFs")
    eval_roi.add_argument("--input-dir", "-i", default="local_data/zeiss")
    eval_roi.add_argument("--output-dir", "-o", default="local_results/auto_roi_campaign")

    sub.add_parser("benchmark", help="Run the bundled deterministic SIMPoly benchmark")

    oracle = sub.add_parser("oracle", help="Manage external scientific oracles")
    oracle_sub = oracle.add_subparsers(dest="oracle_kind", required=True)
    matlab = oracle_sub.add_parser("matlab", help="MATLAB SIMPoly oracle")
    matlab_sub = matlab.add_subparsers(dest="matlab_action", required=True)
    matlab_sub.add_parser("check", help="Check MATLAB batch availability")
    matlab_sub.add_parser("probe", help="Run deterministic morphology probes")
    matlab_run = matlab_sub.add_parser("run", help="Run one source-compatible TIFF case")
    matlab_run.add_argument("--image", required=True)
    matlab_run.add_argument("--output", default=".validation/matlab-oracle/manual-case")
    matlab_run.add_argument("--timeout", type=float, default=1800)

    campaign = sub.add_parser("campaign", help="Run the private 16-TIFF validation campaign")
    campaign_sub = campaign.add_subparsers(dest="campaign_action", required=True)
    campaign_inventory = campaign_sub.add_parser("inventory", help="Freeze the canonical inventory")
    campaign_inventory.add_argument("--dataset", type=Path)
    campaign_run = campaign_sub.add_parser("run", help="Run MATLAB, Python SIMPoly and Fathom")
    campaign_run.add_argument("--methods", default="matlab-simpoly,python-simpoly,fathom")
    campaign_run.add_argument("--case")
    campaign_run.add_argument("--limit", type=int)
    campaign_run.add_argument("--resume", action="store_true")
    campaign_run.add_argument("--force", action="store_true")
    campaign_run.add_argument("--workers", type=int, default=1)
    campaign_run.add_argument("--timeout", type=float, default=900)
    campaign_sub.add_parser("report", help="Build review queue and HTML report")
    unified_campaign = campaign_sub.add_parser(
        "unified", help="Run cached MATLAB/Python/Fathom unified comparison"
    )
    unified_campaign.add_argument(
        "--dataset", type=Path, default=Path(os.environ.get("FATHOM_ZEISS_DATASET", "local_data/zeiss/30-07-26"))
    )
    unified_campaign.add_argument("--matlab-cache-root", type=Path)
    unified_campaign.add_argument("--resume", action="store_true")
    unified_campaign.add_argument("--case", help="Run one canonical case while preserving unified artifacts")
    campaign_sub.add_parser("unified-report", help="Render unified comparison HTML report")

    methods = sub.add_parser("methods", help="List unified scientific method backends")
    methods_sub = methods.add_subparsers(dest="methods_action", required=True)
    methods_sub.add_parser("list", help="List methods and capabilities")

    analyze = sub.add_parser("analyze", help="Run unified methods on one image")
    analyze.add_argument("--image", required=True)
    analyze.add_argument("--methods", default="matlab-simpoly,python-simpoly,fathom-local,fathom-field-graph")
    analyze.add_argument("--matlab-cache-root", type=Path)

    compare = sub.add_parser("compare", help="Compare unified methods on one image")
    compare.add_argument("--image", required=True)
    compare.add_argument("--matlab-cache-root", type=Path)

    args = parser.parse_args()
    if args.command in {None, "gui"}:
        from .ui.app import launch

        raise SystemExit(
            launch(
                getattr(args, "path", None),
                smoke_test=getattr(args, "smoke_test", False),
            )
        )
    if args.command == "inspect":
        raise SystemExit(_inspect(args.paths, args.hash))
    if args.command == "evaluate-auto-roi":
        from scripts.evaluate_auto_roi import run_real_campaign

        res, _inv, _rois = run_real_campaign(Path(args.input_dir), Path(args.output_dir))
        print(f"Informe de evaluación de campaña Auto-ROI: {res}")
        return
    if args.command == "plugins":
        from .plugin_registry import discover_classical, discover_models

        payload = {
            "classical": [
                provider.__dict__
                if hasattr(provider, "__dict__")
                else {
                    "name": provider.name,
                    "group": provider.group,
                    "value": provider.value,
                    "distribution": provider.distribution,
                    "error": provider.error,
                }
                for provider in discover_classical(load=args.load)
            ],
            "models": [
                provider.__dict__
                if hasattr(provider, "__dict__")
                else {
                    "name": provider.name,
                    "group": provider.group,
                    "value": provider.value,
                    "distribution": provider.distribution,
                    "error": provider.error,
                }
                for provider in discover_models(load=args.load)
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "inventory":
        raise SystemExit(_inventory(args.directory, args.output))
    if args.command == "benchmark":
        from .oracles.simpoly import run_synthetic_benchmark_suite

        _runs, _comparisons, summary = run_synthetic_benchmark_suite()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "methods":
        from .core.methods import MethodId

        capability_sets = {
            MethodId.MATLAB_SIMPOLY: ("GLOBAL_DIAMETER_DISTRIBUTION", "LOCAL_EDT_DIAMETERS", "MASK", "SKELETON"),
            MethodId.PYTHON_SIMPOLY: ("GLOBAL_DIAMETER_DISTRIBUTION", "LOCAL_EDT_DIAMETERS", "MASK", "SKELETON", "MATLAB_COMPATIBILITY_EVIDENCE"),
            MethodId.FATHOM_LOCAL: ("LOCAL_METROLOGY", "CROSS_SECTIONS", "MANUAL_REVIEW", "QUALITY_FLAGS"),
            MethodId.FATHOM_FIELD_GRAPH_V1: ("MASK", "SKELETON", "ORIENTATION_FIELD", "LOCAL_RADIUS", "LOCAL_DIAMETER", "GLOBAL_DIAMETER_DISTRIBUTION"),
            MethodId.MANUAL_5X5_REFERENCE: ("MANUAL_REVIEW", "LOCAL_METROLOGY"),
            MethodId.CONSENSUS_PSEUDO_REFERENCE_V1: (),
        }
        print(json.dumps([
            {
                "method_id": method.value,
                "capabilities": capability_sets[method],
                "status": "EXPERIMENTAL_FIELD_MEASURING" if method == MethodId.FATHOM_FIELD_GRAPH_V1 else None,
                "note": "PARTIAL; KNOWN_LIBRARY_DIVERGENCE: bwskel" if method == MethodId.PYTHON_SIMPOLY else None,
            }
            for method in MethodId
        ], indent=2))
        return
    if args.command in {"analyze", "compare"}:
        from .api import FathomEngine

        engine = FathomEngine()
        image = engine.open_image(args.image)
        comparison = engine.compare_all_methods(image, matlab_cache_root=args.matlab_cache_root)
        print(json.dumps({
            "image_id": comparison.image_id,
            "methods": [
                {
                    "method": result.method_id.value,
                    "status": result.status.value,
                    "native_estimand": result.native_estimand.value if result.native_estimand else None,
                    "native_result": result.native_result,
                    "flags": result.quality_flags,
                }
                for result in comparison.results
            ],
            "agreements": [
                {"left": item.left_method.value, "right": item.right_method.value, "wasserstein_1": item.wasserstein_1, "median_difference": item.median_difference}
                for item in comparison.agreements
            ],
            "consensus": {"participating_methods": [item.value for item in comparison.consensus.participating_methods], "excluded_methods": comparison.consensus.excluded_methods, "label": "CONSENSUS_PSEUDO_REFERENCE_V1"},
        }, indent=2, default=str))
        return
    if args.command == "oracle":
        from .validation.matlab_oracle import MatlabOracle

        repo = Path.cwd().resolve()
        matlab_oracle = MatlabOracle.discover(repo)
        if matlab_oracle is None:
            print("MATLAB executable not found", file=sys.stderr)
            raise SystemExit(2)
        if args.matlab_action == "check":
            payload = matlab_oracle.check()
            print(json.dumps(payload, indent=2))
            raise SystemExit(0 if payload["available"] else 1)
        harness = matlab_oracle.harness_dir.as_posix().replace("'", "''")
        if args.matlab_action == "probe":
            output = (repo / ".validation/matlab-oracle/morphology_probes.mat").as_posix()
            completed = matlab_oracle.batch(
                f"addpath('{harness}');run_morphology_probes('{output}');",
                timeout=600,
            )
        else:
            image = Path(args.image).resolve().as_posix().replace("'", "''")
            output = Path(args.output).resolve().as_posix().replace("'", "''")
            completed = matlab_oracle.batch(
                f"addpath('{harness}');run_simpoly_case('{image}','SOURCE_COMPAT',[],"
                f"'{output}',true);",
                timeout=args.timeout,
            )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)
    if args.command == "campaign":
        from .validation.real_campaign import (
            generate_report,
            inventory_dataset,
            run_matlab_campaign,
            run_python_campaign,
        )

        repo = Path.cwd().resolve()
        if args.campaign_action in {"unified", "unified-report"}:
            from .validation.unified_methods import generate_unified_report, run_unified_campaign

            if args.campaign_action == "unified-report":
                print(generate_unified_report(repo))
            else:
                cache_root = args.matlab_cache_root or Path(
                    os.environ.get("FATHOM_MATLAB_CACHE_ROOT", "")
                )
                report = run_unified_campaign(
                    repo,
                    dataset=args.dataset,
                    matlab_cache_root=cache_root if str(cache_root) else None,
                    resume=args.resume,
                    case=args.case,
                )
                print(f"Unified campaign: {len(report.images)} complete, {len(report.failures)} failed")
            return
        if args.campaign_action == "inventory":
            payload = inventory_dataset(repo, args.dataset)
            print(f"Frozen {payload['case_count']} cases for {payload['dataset_id']}")
            return
        if args.campaign_action == "report":
            print(generate_report(repo))
            return
        if args.workers < 1:
            parser.error("--workers must be positive")
        methods = {value.strip() for value in args.methods.split(",") if value.strip()}
        unknown = methods - {
            "matlab-simpoly",
            "python-simpoly",
            "fathom",
            "fathom-local",
            "fathom-field-graph",
        }
        if unknown:
            parser.error(f"unknown methods: {', '.join(sorted(unknown))}")
        # MATLAB is deliberately a single batch session. Python SIMPoly and
        # Fathom share one per-case worker to release image memory on exit.
        if "matlab-simpoly" in methods:
            print(
                f"MATLAB results: {run_matlab_campaign(repo, timeout=max(args.timeout, 900) * 16, force=args.force)}"
            )
        if methods & {"python-simpoly", "fathom", "fathom-local", "fathom-field-graph"}:
            print(
                "Python results: "
                f"{run_python_campaign(repo, case=args.case, limit=args.limit, resume=args.resume, force=args.force, timeout=args.timeout, workers=args.workers)}"
            )


if __name__ == "__main__":
    main()
