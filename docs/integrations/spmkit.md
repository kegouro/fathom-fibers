# SPMKit integration

## Inspected checkout

The adapter was designed against the local checkout:

```text
/home/kegouro/HIBRIS/Workshop ⁄ Proyectos/SPMKIT ALL/spmkit-core-first-audit
commit 149d0e2
```

SPMKit 0.1.5.dev0 exposes two public contracts:

- `spmkit.plugins.v1`: `DatasetInfo`, `Reader`, `Analysis` and `Domain` Protocols;
- `spmkit.gui.modules`: workspace `ModuleSpec`/`PanelSpec` entry points.

Its public `SPMChannel` contains `data`, `unit`, `x_range`, `y_range`, direction,
group and metadata, and computes pixel sizes as range divided by array dimension.

## Adapter boundary

`fathom_fibers_quick.integrations.spmkit.from_spm_channel` accepts that public
structural shape and produces `ScientificImage`. X/Y ranges become explicit
anisotropic calibration. The signal is copied and, by default, robustly normalized
for image segmentation; original signal unit/range and the normalization decision
remain in metadata. No SPMKit object is mutated.

`FATHOM_DOMAIN` satisfies the real runtime `Domain` Protocol and is published under
the `spmkit.plugins.v1` entry-point group. Its `FathomAnalysisProvider` satisfies
the `Analysis` Protocol and delegates `fathom`, `simpoly-controlled` and `compare`
to `FathomEngine`.

```python
from fathom_fibers_quick.integrations.spmkit import FathomAnalysisProvider

provider = FathomAnalysisProvider()
result = provider.run(channel, method="fathom", roi_bbox=(0, 0, 512, 512))
```

## Exact host gap

SPMKit's v1 registry currently registers readers only. When it discovers a
`Domain`, `_register_object` iterates `domain.readers` but does not register or
expose `domain.analyses`. Therefore the adapter is contract-valid and callable,
but automatic analysis-provider discovery is not yet end-to-end in the host.

Fathom does not import private SPMKit registries or build a PyQt6 host panel to
work around that gap. SPMKit should add a public analysis registry/execution and
result/error contract before a host-native UI module is shipped. The standalone
Fathom desktop remains PySide6; mixing its widgets into SPMKit's PyQt6 process is
explicitly avoided.

