# Measurement workflow

1. Open a TIFF and verify calibration, source status and valid image body.
2. Select the measurement protocol and active fiber.
3. Use projected width for approximately perpendicular, clean sections. Avoid
   crossings, image edges, merged regions and the SEM footer unless documenting an
   exclusion.
4. Inspect the physical value, pixel equivalent, calibration snapshot, resolution
   flags and uncertainty in the inspector.
5. Use 3–5 accepted sections per fiber when the selected protocol requires them.
6. Keep assisted Fathom and SIMPoly outputs as `PROPOSED` until visual review.
7. Accept, reject or mark ambiguous; only `ACCEPTED` and `MANUALLY_EDITED` enter
   primary summaries.
8. Save the project explicitly. Autosave is a separate recovery copy, not the
   authoritative project.
9. Export the reviewed selection or complete project with provenance.

Viewer brightness, contrast, gamma and inversion affect display only. The raw
array used for measurements and analysis is never modified.

SIMPoly Source Compatible follows the source's fixed 90-row crop. SIMPoly
Controlled Input and Fathom can share a caller-selected ROI for method comparison.
The source-compatible pixel domain should be treated as approximately 10–100 px;
out-of-domain results are flagged. Do not label either method as ground truth.

All results describe projected 2D geometry. A projected width becomes a physical
fiber diameter only under additional geometric/material assumptions that the
software does not establish.

