# Fathom Fibers User Guide

## 1. What Fathom Fibers does

Fathom Fibers is a scientific desktop workspace for measuring fiber diameters in
SEM images. You open a folder of SEM images, the application runs several
independent diameter-analysis methods, and you inspect the results as
distributions, per-image diagnostics, overlays on the image, and a final HTML
scientific report.

The workspace has four top-level modes — **Analyze**, **Manual 5×5**,
**Report** and **Advanced** — selected in the top bar (or with `A`, `M`, `R`).

## 2. Five-minute Quick Start

1. Launch Fathom Fibers.
2. Click **Open Dataset** and select the folder containing your SEM images.
3. Check the dataset status in the left panel ("16 images · analysis available n / 16").
   - If analysis already exists: click **Explore Results**.
   - If results are missing: use **Run missing**.
4. Select an image in the navigator.
5. Inspect the bottom tabs: **Distribution**, **Comparison**, **Quality**.
6. Use **Layers** to overlay centerlines and paired edges on the image if you want geometry feedback.
7. Optional: press `M` and complete the **Manual 5×5** reference.
8. Press `R` and click **Generate Dataset Scientific Report**.
9. Click **Export Analysis Bundle** to save results, figures and the report to a folder.

## 3. The normal workflow

```
Open Dataset
  → Analyze (load or run methods)
  → inspect Distribution / Comparison / Quality
  → optional: Manual 5×5 reference
  → Report (generate the scientific report)
  → Export Analysis Bundle
```

## 4. Analyze workspace

This is the default view after opening a dataset: the SEM image is the central
instrument, with the dataset navigator on the left, the image/selection summary
on the right, and three bottom tabs:

- **Distribution** — histograms and ECDFs of measured diameters per method.
- **Comparison** — method-by-method agreement tables.
- **Quality** — coverage, acceptance, flags and refinement diagnostics.

The **DATASET** panel shows every image with a status icon:

| Icon | Meaning |
| ---- | ------- |
| ✓ complete | cached analysis available |
| summary cache | summary-only results available |
| not analyzed | nothing computed yet |
| running | analysis in progress |

The header shows the dataset summary and one primary action:

- **Explore Results** — all images already have cached analysis.
- **Analyze Dataset** — some images still need analysis (equivalent to Run missing).

### The three run actions

| Action | What it does | When to use |
| ------ | ------------ | ----------- |
| **Run Methods** | Runs the analysis methods for the current image (dialog lets you choose which). | Targeted work on one image. |
| **Run missing** | Analyzes only the dataset images that do not yet have a valid cached result. | Normal continuation — preserves existing results. |
| **Run all dataset** | Recomputes the analysis for the whole dataset, including images that already have results. | Deliberately refreshing everything; more expensive. Use intentionally, not by default. |

Results are cached per image; switching images or modes never reruns anything.

## 5. Manual 5×5 workspace

Manual 5×5 is a focused measurement environment for collecting a sparse human
reference: 25 positions per image (a 5×5 grid), 400 measurements for a
16-image dataset.

1. Press `M` (or click **Manual** in the top bar).
2. The current target is highlighted and the view zooms to it.
3. Identify the fiber associated with the target and draw a perpendicular
   **projected-width line** (`M` activates the tool; click-drag across the fiber).
4. The measurement is accepted and **autosaved immediately** — you see a brief
   `Saved ✓` confirmation.
5. `Enter` — next target · `Backspace` — previous target · `Delete` — remove
   the current measurement · `Esc` — cancel the line being drawn.
6. Continue until the grid shows 25 / 25.

The bottom grid shows the 25 targets with distinct marks:

| Mark | Meaning |
| ---- | ------- |
| measured | a width was recorded |
| current | the active target |
| pending | not yet visited |
| skipped | intentionally skipped with a reason |

Manual 5×5 is a **sparse human reference, not automatic ground truth**.
Missing measurements are never invented or filled in.

## 6. Report workspace

Press `R` to enter Report mode. The header offers:

- **Generate Dataset Scientific Report** — the main final analysis of the full
  dataset (HTML: dataset overview, image-level method summary, Oriented Ribbon
  dataset behavior, figures, quality, Manual 5×5, methods, per-image sections,
  provenance, limitations). Per-image reports lead with a Scientific Summary,
  key diameter results, a Raw-vs-Ribbon centerline refinement table, primary and
  full-range distributions, agreement, quality and limitations.
- **Export Analysis Bundle** — a portable results package (see below).
- **Current Image Report** — a report for the single selected image.

## 7. Advanced workspace

Advanced exposes technical diagnostics for expert use: Methods, Measurements,
History, Analysis, Batch Measurement Review, project internals and full
diagnostic panels. Normal dataset analysis does not require Advanced.

## 8. Methods explained

| Method | Meaning | Status |
| ------ | ------- | ------ |
| **MATLAB SIMPoly** | Reference/native implementation, consumed from a validated cache. The cache reports the native Gaussian center **b1**; a full diameter distribution is not available, so no histogram/ECDF is fabricated for it. | COMPLETE (cache) |
| **Python SIMPoly** | Python source-compatible approximation of SIMPoly (calibrated length-weighted diameters on the skeleton). | COMPLETE |
| **Fathom Local** | Independent local cross-section estimator on fiber candidates. It samples cross-sections differently and may produce a broader distribution than the Field methods; it is not truth. | COMPLETE |
| **Fathom Field (Raw)** | Structure-tensor orientation plus local boundary metrology on the current centerline: **Raw EDT**, **Raw Edge**, **Raw Profile** (see below). | EXPERIMENTAL |
| **Oriented Ribbon V1** | Experimental centerline refinement that re-measures the same estimators along a refined geometric centerline: **Ribbon EDT / Edge / Profile**. | EXPERIMENTAL |
| **Manual 5×5** | Sparse human reference grid. | REFERENCE |
| **Consensus** | A pseudo-reference formed from the participating methods' quantiles (Python SIMPoly, Fathom Local, Fathom Field). Methods without a comparable distribution (MATLAB b1-only, Manual) are excluded. Field Raw/Ribbon are estimator variants of one method and do **not** add independent votes. **Not ground truth.** | REFERENCE |

### Field terminology

- **Raw EDT** — twice the physical distance from the current sampled centerline to the nearest mask boundary.
- **Raw Edge** — distance between both paired local boundaries measured along the local fiber normal.
- **Raw Profile** — the paired-edge width refined against the raw SEM intensity profile.
- **Ribbon EDT / Ribbon Edge / Ribbon Profile** — the same three estimators, re-measured after Oriented Ribbon V1 centerline refinement.

### Oriented Ribbon V1

**EXPERIMENTAL.** The mechanism — geometric midpoints from paired opposite
boundaries, a confidence-weighted smooth centerline on non-branching runs, then
re-measurement — is validated on known-truth synthetic geometry. Real SEM
comparisons show method behavior and agreement, **not known absolute accuracy**.
Ribbon coverage below 100% is not a failure: unsupported samples include
deliberately abstained regions such as crossings, junctions, gaps and
low-confidence geometry. Abstention is preferable to inventing a measurement.

## 9. Distribution plots

- **Histogram** — weighted density of measured diameters; where diameters are concentrated.
- **ECDF** — the fraction of (weighted) measurements below a given diameter.

All curves share the same physical bins and units (µm). Presets:

| Preset | Shows |
| ------ | ----- |
| ALL METHODS | every available series |
| FIELD RAW | Raw EDT / Edge / Profile |
| FIELD RIBBON | Ribbon EDT / Edge / Profile |
| RAW vs REFINED EDT | Raw EDT vs Ribbon EDT |
| RAW vs REFINED EDGE | Raw Edge vs Ribbon Edge |
| RAW vs REFINED PROFILE | Raw Profile vs Ribbon Profile |
| MANUAL COMPARISON | Manual vs Python SIMPoly vs Fathom Local vs Ribbon estimates |

## 10. Layers / overlays

The **OVERLAYS** panel toggles what is drawn on the image:

| Layer | What you are looking at |
| ----- | ----------------------- |
| Manual measurements | your recorded width lines |
| Fathom Local measurements | local cross-sections on fiber candidates |
| Python SIMPoly skeleton | the skeleton used by Python SIMPoly |
| Python SIMPoly mask | the segmentation mask |
| Field centerline | the skeleton centerline used by the Field |
| Raw centerline | the sampled centerline positions |
| Refined centerline | the Oriented Ribbon smooth centerline (gaps are intentional) |
| Midpoint observations | geometric midpoints between paired boundaries |
| Field orientation | local fiber-axis orientation ticks |
| Raw paired-edge segments | paired mask-boundary width lines |
| Refined paired-edge segments | paired boundaries re-measured after refinement |
| Field profile-refined edges | intensity-profile refined edge positions |
| Rejected / flagged samples | samples that failed acceptance |
| Rejected refinement samples | samples where the refinement abstained |

Layers that have no data for the current image are hidden, not shown disabled.
**Orientation density** (Sparse / Medium / Dense) changes display density only —
it never changes the science.

## 11. Inspector

The right panel shows a compact **Image** summary when nothing is selected:
current image, calibration, median values for the Field estimators, coverage
and acceptance. Clicking a measurement or sample switches it to the detailed
views:

- **Measurement / Workspace** — the selected sample:
  - original and refined center positions
  - observed center shift and residual shift
  - raw vs refined widths (EDT / Edge / Profile, r− / r+, asymmetry)
  - coherence, confidence, orientation disagreement, flags
- **Display** — brightness/contrast/gamma controls (display only).

Units are always shown explicitly.

## 12. Quality indicators

- **Coverage** — fraction of samples with a supported analysis.
- **Acceptance** — fraction of samples whose edges/profiles were accepted.
- **Flags** — per-sample diagnostic flags with counts.
- **Coherence** — local orientation quality.
- **Refinement support** — which samples the Ribbon centerline covers.

Ribbon coverage below 100% is expected and intentional (abstention on
crossings, junctions, gaps, low-confidence regions).

## 13. Reports and exports

- **Generate Dataset Scientific Report** → `report/index.html` — the deliverable for a professor/collaborator.
- **Generate Current Image Report** → a single-image report.
- **Export Analysis Bundle** → a portable folder:

| Path | Contents |
| ---- | -------- |
| `report/index.html` | the full scientific report (primary human-readable file) |
| `results/dataset_summary.csv` | one row per image × method × estimator (median, IQR, P05/P95, N, status) |
| `results/measurements.csv` | tidy per-sample measurements (may be very large) |
| `results/manual_5x5.csv` | manual grid records when operator measurements exist |
| `results/method_results.json` | full method results and distribution summaries |
| `results/provenance.json` | versions, hashes, calibration, cache provenance |
| `figures/` | dataset and per-image figures |

To send to another person: the primary file is `report/index.html`, the
numerical summary is `results/dataset_summary.csv`, the full local measurements
are `results/measurements.csv` (large), and the portable application is the
packaged FathomFibers distribution. Raw SEM images are intentionally not
included in the bundle.

## 14. Keyboard shortcuts

| Key | Action |
| --- | ------ |
| `A` | Analyze mode |
| `M` | Manual 5×5 mode (also selects the projected-width tool while measuring) |
| `R` | Report mode |
| `Left` / `Right` | Previous / next image (outside measurement drawing) |
| `F` | Fit image |
| `1` | 1:1 pixels |
| `0` | Reset view |
| `Ctrl+O` | Open image |
| `Ctrl+S` / `Ctrl+Shift+S` | Save / Save project as |
| `Ctrl+E` | Export CSV |
| `Ctrl+R` | Generate current image report |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
| `Delete` | Delete selected measurement |

Manual 5×5 mode: `Enter` accept/next · `Backspace` previous · `Delete` remove ·
`Esc` cancel the line being drawn.

## 15. Common questions / troubleshooting

- **Why does the dataset show "Run missing"?** Some images do not have a valid cached analysis yet; Run missing only fills those.
- **Why is nothing computed when I switch images?** Results are cached; switching never reruns. Use Run missing / Run methods when a result is genuinely missing.
- **Why is there no MATLAB histogram?** The MATLAB cache reports the native b1 statistic but not a full diameter array; fabricating one would be unscientific.
- **Why is Ribbon coverage below 100%?** Intentional abstention on ambiguous regions.
- **Why are some overlays missing from the Layers panel?** Layers with no data for the current image are hidden.
- **The analysis seems to hang?** Long operations show their status in the bottom status bar (e.g., "Analyzing image 4 / 16"). Reports and bundles also run in the background.
- **Where do my manual measurements go?** They are autosaved per dataset immediately and appear in the Manual 5×5 grid and the report.
