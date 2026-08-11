# Local MATLAB oracle harness

The executable harness lives under `.validation/matlab-oracle/src/` and is
intentionally ignored: it contains a close procedural transcription of the
externally supplied MATLAB supplement. The canonical source in `Downloads` and
the ignored reference copy are never modified or packaged.

Versioned Python code in `fathom_fibers_quick.validation` discovers MATLAB,
calculates cache keys, invokes this harness, parses MAT/JSON results and computes
parity metrics. The core and public headless API do not import it.

Local commands:

```bash
fathom-fibers oracle matlab check
fathom-fibers oracle matlab probe
fathom-fibers oracle matlab run --image image.tif
fathom-fibers campaign inventory
fathom-fibers campaign run --methods matlab-simpoly,python-simpoly,fathom --resume
fathom-fibers campaign report
```

`FATHOM_MATLAB_EXECUTABLE` overrides executable discovery. Do not copy license
tokens, host IDs, private TIFFs, source supplements or `.validation` outputs into
Git.
