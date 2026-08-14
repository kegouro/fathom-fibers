# Current status

The default application is the PySide6 scientific workspace. The Qt-free
`FathomEngine` and application `ProjectSession` are shared by the desktop UI and
SPMKit adapter. The legacy Tk UI is preserved at `fathom_fibers_quick.app` but is
not the default entrypoint.

Verified locally on 2026-08-12:

- unified scientific workspace: dataset navigation (16 images), cached
  `MethodResult` loading, background method runs with per-stage progress,
  weighted histogram + ECDF comparison, field sample review with bidirectional
  selection, quality flag breakdown, overlay layers (mask, skeleton, centerline,
  orientation, paired edges, profile edges, rejected samples), manual 5×5
  workflow with per-acceptance autosave, image and dataset HTML reports and
  CSV/JSON exports;
- workspace full-result cache round-trips arrays, secondary distributions and
  per-sample flags exactly; summary-only campaign runs remain readable;
- MATLAB SIMPoly is consumed from the validated cache (b1 native Gaussian only;
  no raw arrays); the workspace never launches MATLAB;
- real-data sanity on the 16-image corpus: no footer measurements, calibration
  5.204e-08 m/px, physical (µm) units, no x/y swap, image navigation and partial
  method states verified;
- canonical SIMPoly source hash matched the supplied SHA-256;
- core imports no Qt and no SPMKit (`workspace` and `reports` modules included);
- Qt shell, viewer, tool measurement, table/inspector synchronization,
  undo/redo, save/reopen and automatic `PROPOSED` results passed offscreen tests;
- SPMKit adapter satisfied the checkout's public runtime `Domain` and `Analysis`
  Protocols;
- SPMKit's registry still lacks end-to-end analysis-provider registration.

This status does not claim production readiness, ground truth or exact MATLAB
parity. See `docs/architecture.md` and the SIMPoly validation profile for exact
limitations.
