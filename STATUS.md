# Current status

The default application is the PySide6 scientific workspace. The Qt-free
`FathomEngine` and application `ProjectSession` are shared by the desktop UI and
SPMKit adapter. The legacy Tk UI is preserved at `fathom_fibers_quick.app` but is
not the default entrypoint.

Verified locally on 2026-08-11:

- canonical SIMPoly source hash matched the supplied SHA-256;
- core imports no Qt and no SPMKit;
- Qt shell, viewer, tool measurement, table/inspector synchronization,
  undo/redo, save/reopen and automatic `PROPOSED` results passed offscreen tests;
- real Zeiss smoke completed on `PVDF Jose_02.tif`, `_09.tif` and `_13.tif`;
- SPMKit adapter satisfied the checkout's public runtime `Domain` and `Analysis`
  Protocols;
- SPMKit's registry still lacks end-to-end analysis-provider registration.

This status does not claim production readiness, ground truth or exact MATLAB
parity. See `docs/architecture.md` and the SIMPoly validation profile for exact
limitations.
