# Fathom Fibers

Fathom Fibers is a scientific desktop workspace and headless Python library for
projected 2D fiber measurements in microscopy images. It reads Zeiss SEM TIFF
calibration, supports manual and reviewable assisted workflows, and exposes the
same scientific engine to the PySide6 application and SPMKit adapter.

> Measurements represent projected 2D geometry. Automatic results enter as
> `PROPOSED` and are excluded from primary statistics until reviewed.

## Release artifacts

Prebuilt, portable desktop releases (no Python installation required) are
provided for Linux, Windows and macOS:

`FathomFibers-0.2.0-rc1-<platform>-<arch>.tar.gz` / `.zip`

Each archive contains the application, `README_FIRST.md`, `LICENSE`,
`CHANGELOG.md` and `VERSION`. Scientific status: Python SIMPoly is a
source-compatible approximation validated against MATLAB with a known library
divergence; Fathom Field and Oriented Ribbon V1 are experimental; Manual 5×5
is a sparse human reference; Consensus is a pseudo-reference. Real SEM results
characterize agreement, not known absolute accuracy.

## Start in under two minutes

```bash
git clone <repository-url> fathom-fibers
cd fathom-fibers
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gui]"
fathom-fibers
```

Development setup:

```bash
python -m pip install -e ".[gui,dev,validation]"
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

On Arch Linux, install Python and Qt's common runtime libraries first if needed:

```bash
sudo pacman -S --needed python base-devel libglvnd
```

The repository launcher uses `.venv`, writes logs below
`/tmp/fathom-fibers-$USER/`, and can run checks before launch:

```bash
./scripts/run-fathom.sh
./scripts/run-fathom.sh --check
```

## Desktop workflow

The Qt workspace has a project hierarchy, central scientific viewer, inspector,
and bottom results/history/analysis/comparison panels. The viewer provides wheel
zoom, pan, fit, 1:1 pixels, coordinates, raw pixel value, physical coordinates,
scale bar, valid-image body/footer inspection, and non-destructive brightness,
contrast, gamma and inversion.

Tools and shortcuts:

| Tool/action | Shortcut |
|---|---|
| Select | `V` |
| Pan | `H` or hold `Space` |
| Projected width | `M` |
| Distance | `D` |
| Polyline | `P`, `Enter` to finish |
| Angle | `G` |
| Rectangle ROI | `R` |
| Polygon ROI | `Y`, `Enter` to finish |
| Intensity profile | `L` |
| Cancel | `Esc` |
| Delete | `Delete` |
| Undo / redo | `Ctrl+Z` / `Ctrl+Shift+Z` or `Ctrl+Y` |
| Fit / 1:1 | `F` / `1` |

See [measurement workflow](docs/measurement-workflow.md) for review rules and
estimand cautions.

## Scientific workspace

From one window the workspace opens a dataset, navigates all 16 images, loads or
runs the unified methods, compares distributions, reviews individual field
measurements and completes the manual 5×5 protocol.

1. **Launch** — `fathom-fibers` (or `./launch.py`).
2. **Open dataset** — toolbar or *File → Open dataset*; choose the folder
   containing your SEM TIFF images. The DATASET dock lists the images with
   `complete / summary cache / not analyzed` status and per-image manual progress.
3. **Run methods** — `R` opens the run dialog: Python SIMPoly, Fathom Local and
   Fathom Field run in background workers; MATLAB is consumed from the validated
   cache (never launched). Full results are cached per image under
   `.validation/unified-method-comparison/full/`, so revisiting an image never
   reruns algorithms. `Run missing` / `Run all dataset` fill the cache ahead of
   time (`scripts/cache_workspace_results.py` does the same headless).
4. **View distributions** — the DISTRIBUTIONS tab shows a weighted density
   histogram and ECDF with identical bins and units across series, plus pairwise
   Wasserstein-1 / KS / median-difference tables. Field EDT, Paired Edge and
   Intensity Profile are estimator variants of one experimental method.
5. **Manual 5×5** — the MANUAL 5×5 tab guides 25 targets per image: next target,
   draw a perpendicular width line (`M`), measurement is accepted and autosaved
   immediately. `Enter` next, `Backspace` previous, `Delete` remove, `Esc`
   cancel. Progress is persisted per dataset (`.validation/…/manual5x5/`) so a
   crash never loses the 400-measurement campaign.
6. **Generate report** — *Report → Generate scientific report* (image,
   `Ctrl+R`) or *Generate dataset scientific report*; both are headless HTML
   reports with figures, written under `.validation/unified-method-comparison/`.
7. **Export** — current-image CSV/JSON or dataset summary CSV.

The UI contains no scientific algorithms: Qt widgets talk to the
`WorkspaceController`, which orchestrates the Qt-free `workspace` layer,
`FathomEngine`, `ProjectSession` and `MethodResult` contracts. The scientific
core remains fully usable headless and never imports PySide6.

## Headless API

```python
from fathom_fibers_quick.api import FathomEngine

engine = FathomEngine()
image = engine.open_image("micrograph.tif")

line = engine.measure(
    image,
    "PROJECTED_WIDTH",
    {"p1": (120.0, 80.0), "p2": (120.0, 112.0)},
)
fathom = engine.run_fathom(image, roi_bbox=(100, 60, 500, 460))
simpoly, intermediates = engine.run_simpoly(
    image,
    profile="SIMPOLY_CONTROLLED_INPUT_V1",
    roi_bbox=(100, 60, 500, 460),
)
comparison = engine.compare_methods(image, roi_bbox=(100, 60, 500, 460))
unified = engine.compare_all_methods(image)
```

Arrays are supported without filesystem access; calibration is always explicit.
Full examples and contracts are in [docs/api.md](docs/api.md).

## CLI

```bash
fathom-fibers                         # opens Qt
fathom-fibers gui [image-or-project]
fathom-fibers inspect --hash image.tif
fathom-fibers inventory images/ -o inventory.csv
fathom-fibers benchmark
fathom-fibers oracle matlab check
fathom-fibers oracle matlab probe
fathom-fibers campaign inventory
fathom-fibers campaign run --methods matlab-simpoly,python-simpoly,fathom --resume
fathom-fibers campaign report
fathom-fibers methods list
fathom-fibers compare --image image.tif --matlab-cache-root /path/to/.validation/real-tiff-campaign
fathom-fibers campaign unified --dataset /path/to/16-tiffs --matlab-cache-root /path/to/.validation/real-tiff-campaign
fathom-fibers campaign unified-report
```

MATLAB and the private TIFF corpus are optional validation adapters. The normal
suite does not require either; licensed runtime tests are enabled explicitly with
`FATHOM_MATLAB_EXECUTABLE=/path/to/matlab pytest -m matlab`.

An offscreen-safe shell smoke test is available:

```bash
QT_QPA_PLATFORM=offscreen fathom-fibers gui --smoke-test
```

## Scientific scope

Fathom preserves Zeiss metadata, anisotropic calibration, measurement geometry,
protocol snapshots, uncertainty, repeatability, hierarchical statistics,
provenance, source hashes, autosave, undo/redo and atomic project persistence.
The source-compatible SIMPoly port retains the source's algorithm order and
literal decisions. R2026a probes establish parity for selected operations, but
CLAHE, Canny, complex thickening, skeletonization, automatic histogram binning
and nonlinear fitting are not claimed as exact MATLAB parity.
See [SIMPoly validation profile](docs/validation/simpoly-source-profile.md).

## Unified method comparison

`FathomEngine.compare_all_methods()` adapts MATLAB SIMPoly cache results, Python
SIMPoly, Fathom Local, manual records and the registered Field Graph backend to
one typed `MethodResult`. The common comparison estimand is
`COMMON_LENGTH_WEIGHTED_DIAMETER`; native estimands remain visible separately.
The optional `CONSENSUS_PSEUDO_REFERENCE_V1` is a median quantile curve with
equal method weight. It is never called ground truth.

`FATHOM_FIELD_GRAPH_V1` is an experimental field-measuring backend: orientation
field, anisotropic EDT radii and paired-edge metrology with intensity-profile
refinement are implemented and reported with `EXPERIMENTAL_FIELD_MEASURING`
status. Graph reconstruction, crossing resolution and fiber instances are not
implemented and are not hidden. Future Omnipose, instance embedding and ML
perception backends use the Qt-free `FiberPerceptionBackend` contract; no model
runtime is installed.

The old Tk application remains at `fathom_fibers_quick.app` for migration safety;
the default entrypoint now launches Qt. Scientific code does not import Qt or
SPMKit. Architecture and remaining limitations are documented in
[docs/architecture.md](docs/architecture.md).
