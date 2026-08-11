# Fathom Fibers

Fathom Fibers is a scientific desktop workspace and headless Python library for
projected 2D fiber measurements in microscopy images. It reads Zeiss SEM TIFF
calibration, supports manual and reviewable assisted workflows, and exposes the
same scientific engine to the PySide6 application and SPMKit adapter.

> Measurements represent projected 2D geometry. Automatic results enter as
> `PROPOSED` and are excluded from primary statistics until reviewed.

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

The old Tk application remains at `fathom_fibers_quick.app` for migration safety;
the default entrypoint now launches Qt. Scientific code does not import Qt or
SPMKit. Architecture and remaining limitations are documented in
[docs/architecture.md](docs/architecture.md).
