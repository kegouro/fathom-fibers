# Architecture

## Boundary

```text
FATHOM CORE
    ↓
APPLICATION / USE CASES
    ↓
┌──────────────────────┬──────────────────────┐
│ Standalone PySide6 UI │ SPMKit adapter       │
└──────────────────────┴──────────────────────┘

External validation is a sibling adapter, never an upstream core dependency:

```text
MATLAB R2026a oracle → validation/parity contracts → application/report/Qt review
```
```

`fathom_fibers_quick.core` and the existing scientific modules import neither Qt
nor SPMKit. `FathomEngine` is the stable façade over array/image loading,
measurement geometry, Fathom assisted ROI, both SIMPoly profiles and method
comparison. `ProjectSession` owns use cases, selection, dirty state and command
history. Frontends only submit requests and render results.

## Initial repository state (2026-08-11)

- Branch: `main`; initial HEAD `7e40bce`.
- Working tree already contained an untracked private image ZIP and an untracked
  launcher. Both were preserved; the ZIP is ignored and the compatible launcher
  is now tracked.
- Baseline: 76 tests passed via `python -m pytest`; direct `pytest` failed to import
  the namespace package `scripts`. Adding repository root to pytest's python path
  makes both invocations equivalent.
- `ruff` and `compileall` passed.
- Scientific/backend modules: Zeiss I/O, project I/O, geometry, analysis,
  auto-ROI, SIMPoly ports/oracles, protocols, validation, hierarchy,
  repeatability, provenance, exporters, autosave and history.
- GUI: one 1,346-line Tk class in `app.py`, coupling widgets, dialogs, mutable UI
  state, domain records, analysis, persistence, exports and history.
- Project schema v3 stored one image plus typed records and backward migration;
  autosave existed but its temporary suffix path could produce an empty recovery
  file. Schema v4 adds ROI/analysis/history metadata, reliable atomic writes and
  non-destructive legacy backups.

## Current modules

| Layer | Modules | Responsibility |
|---|---|---|
| Core contracts | `core/`, `model.py`, `measurement_records.py` | Arrays, calibration, images, records and typed results. |
| Algorithms | `analysis.py`, `auto_roi.py`, `measurement_geometry.py`, `simpoly_compat.py`, `oracles/` | Numerical and scientific behavior. |
| I/O | `zeiss.py`, `project_io.py`, `exporters.py`, `autosave.py` | File formats, atomic persistence and export. |
| Application | `application/session.py`, `api.py`, `history.py` | Use cases, commands, dirty state, review transitions and headless façade. |
| Qt | `ui/` | Viewer, tools, overlays, model/view panels, dialogs and background tasks. |
| Integration | `integrations/spmkit/` | Public SPMKit channel/domain translation only. |
| Validation adapters | `validation/` | MATLAB process discovery, cache keys, parity metrics, private campaign and review queue. |
| Legacy | `app.py` | Preserved Tk application, no longer the default entrypoint. |

## Legacy behavior map

| UI behavior | Domain operation | Previous location | Migration decision |
|---|---|---|---|
| Open Zeiss TIFF/calibration/footer | `zeiss.load_image_document` | Tk `open_path` | Kept in core; new Qt use case. |
| Zoom/pan/fit/coordinates | display only | Tk canvas methods | Replaced by `ScientificImageView`. |
| Projected width/distance/polyline/angle/area/profile | `measurement_geometry` | Tk callbacks | Kept; Qt tools submit typed geometry. |
| Fiber grouping/protocols | records/protocols | Tk mutable vars | Kept in project/session and model/view. |
| Table/status/tags/notes | `MeasurementRecord` | Tk trees/dialogs | Replaced by table model and inspector. |
| History/undo/redo | `HistoryManager` | Tk closures | Kept behind `HistoryBridge`/session commands. |
| Autosave/project persistence | autosave/project I/O | Tk timers/dialogs | Kept and hardened; Qt timer is non-modal. |
| Automatic ROI | `auto_roi.analyze_roi` | synchronous Tk callback | Kept; Qt worker and `PROPOSED` records. |
| SIMPoly | oracle pipeline | not integrated in Tk | Added to Qt analysis and comparison. |
| Export/reliability/provenance | backend modules | Tk dialogs | Backend kept; CSV exposed in Qt; advanced report/reliability UI deferred. |

## Project format

Schema v4 remains JSON and reads the old `measurements` representation and schema
v1–v3 records. Before overwriting an older project, `save_project` writes a sibling
`*.schema-vN.bak` once. Writes use a same-directory temporary, flush, `fsync`, atomic
replace and directory sync where supported. Autosaves are separate in the user data
directory and never change the explicit project path.

Preserved data include measurements, fibers, ROI IDs/definitions, protocol
snapshots, status, tags, notes, history metadata, calibration, source hashes,
uncertainty and provenance-like analysis run metadata. The current `Project` still
owns one image; the Qt tree presents the intended hierarchy but multi-image project
storage is deferred.

## Feature preservation matrix

| Capability | Initial backend | Current core | Qt exposure |
|---|---:|---:|---:|
| Zeiss metadata/footer/calibration | yes | yes | yes |
| Anisotropic pixels | yes | yes | yes |
| Width/distance/polyline/angle/area/profile | yes | yes | yes |
| Protocol/uncertainty/repeatability/hierarchy | yes | yes | protocol + inspector; study UI deferred |
| Provenance/source hash | yes | yes | inspector/project verification |
| Autosave/undo/redo | yes | hardened | yes |
| Automatic ROI | yes | yes | async, review proposals |
| SIMPoly source/controlled profiles | yes | audited | async + inspector |
| Method comparison | partial script | typed API | table + four previews |
| CSV/annotated/HTML export | yes | yes | CSV; annotated/HTML dialogs deferred |

## Known limitations

- Qt line overlays can be translated with undo/redo; independent endpoint handles
  and editable polygon vertices remain a next-batch UI enhancement.
- Project storage is still single-image even though the tree models the future
  Project/Sample/Image hierarchy.
- Reliability, repeatability and HTML/annotated export backends remain functional,
  but their dedicated Qt dialogs have not yet superseded legacy UI.
- Background cancellation is cooperative at task boundaries; the current numerical
  kernels do not expose fine-grained progress/cancellation callbacks.
- SIMPoly is now cross-validated against MATLAB R2026a. CLAHE, Canny, complex
  thickening, `bwskel`, spur, automatic histogram and fitting retain measured
  divergences; see the validation profile. MATLAB remains optional and never
  enters `core`, `api.FathomEngine` or SPMKit integration.
