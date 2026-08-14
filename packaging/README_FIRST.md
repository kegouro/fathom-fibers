# Fathom Fibers — 5-minute start

Fathom Fibers is a scientific desktop workspace for measuring fiber diameters in
SEM images. No Python installation is needed.

## Start

1. **Start the application**
   - Linux: extract the archive, then `./app/FathomFibers` (if the executable
     bit is lost: `chmod +x app/FathomFibers`).
   - Windows: extract the ZIP, then run `app\FathomFibers.exe`. Windows
     SmartScreen may show a warning for unsigned binaries; verify the file hash
     before running.
   - macOS: extract the ZIP, then open `app/FathomFibers.app`. The application
     is not signed or notarized; macOS may require the standard "Open Anyway"
     flow for applications from an unidentified developer (System Settings →
     Privacy & Security). Do not disable Gatekeeper globally.
2. **Open Dataset** — choose the folder containing your SEM TIFF images.
3. **Analyze Dataset** — if some images still need analysis, the header shows
   *Analyze Dataset* (equivalent to **Run missing**). **Run all dataset**
   recomputes everything and is only needed when deliberately refreshing.
4. **Inspect** the bottom tabs: **Distribution** (histograms and ECDFs),
   **Comparison** (agreement tables), **Quality** (coverage, acceptance, flags).
5. **Optional: Manual 5×5** (`M`) — draw perpendicular width lines on the
   25 targets; measurements autosave immediately.
6. **Report** (`R`) — click **Generate Dataset Scientific Report**.
7. **Export Analysis Bundle** — saves the HTML report, figures, CSV/JSON results
   and provenance to a folder.

Outputs you receive: `report/index.html` (the main report), `figures/`,
`results/dataset_summary.csv`, `results/measurements.csv`,
`results/method_results.json` and `results/provenance.json`.

## Scientific notes

- Measurements are projected 2-D diameters.
- Automatic agreement is not ground truth.
- **Oriented Ribbon V1 is experimental**: known-truth synthetic geometry
  validates the centerline mechanism; real SEM comparisons characterize
  behavior/agreement, not known absolute accuracy.
- Manual 5×5 is a sparse human reference, not ground truth.
- Verify your calibration and image metadata — diameter values are physical µm
  only when calibration is correct.

The full user guide is in `docs/USER_GUIDE.md` inside the repository and via
the in-app Help menu (Quick Start, User Guide, Methods Guide, Keyboard
Shortcuts).
