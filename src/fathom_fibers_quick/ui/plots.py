"""Matplotlib canvases used inside the workspace; no global pyplot state."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..core.distributions import common_histogram_edges, summarize_distribution
from ..core.methods import DiameterDistribution

SERIES_COLORS = {
    "Python SIMPoly": "#fdb462",
    "Fathom Local": "#7fc97f",
    "Fathom Field (EDT)": "#8c6bb1",
    "Field Paired Edge": "#d95f02",
    "Field Intensity Profile": "#1b9e77",
    "Ribbon Refined EDT": "#386cb0",
    "Ribbon Refined Edge": "#4daf4a",
    "Ribbon Refined Profile": "#984ea3",
    "Manual 5×5": "#e31a1c",
    "Consensus": "#252525",
    "MATLAB SIMPoly": "#386cb0",
}


def _weighted_density(distribution: DiameterDistribution, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(distribution.diameter, bins=edges, weights=distribution.weight)
    widths = np.diff(edges)
    total = distribution.weight.sum()
    return hist / (total * widths) if total > 0 else np.zeros_like(hist)


class DistributionCanvas(FigureCanvasQTAgg):
    """Weighted density histogram with a shared bin grid across series."""

    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(6.4, 4.4), dpi=100)
        super().__init__(self.figure)
        self.setParent(parent)
        self._axis = self.figure.add_subplot(111)
        self._message = self._axis.text(
            0.5,
            0.5,
            "No distributions available",
            ha="center",
            va="center",
            transform=self._axis.transAxes,
        )

    def set_series(self, series: list[tuple[str, DiameterDistribution]]) -> None:
        self._axis.clear()
        self._axis.grid(alpha=0.2)
        if not series:
            self._axis.text(
                0.5,
                0.5,
                "No distributions available",
                ha="center",
                va="center",
                transform=self._axis.transAxes,
            )
            self._axis.set_title("Common diameter histogram")
            self.draw_idle()
            return
        distributions = [item[1] for item in series]
        edges = common_histogram_edges(distributions)
        if not edges.size:
            self._axis.text(
                0.5,
                0.5,
                "No comparable distributions",
                ha="center",
                va="center",
                transform=self._axis.transAxes,
            )
        else:
            centers = edges[:-1] + np.diff(edges) / 2.0
            for name, distribution in series:
                self._axis.bar(
                    centers * 1000.0,
                    _weighted_density(distribution, edges) / 1000.0,
                    width=np.diff(edges) * 1000.0,
                    color=SERIES_COLORS.get(name),
                    alpha=0.55,
                    edgecolor="none",
                    align="center",
                    label=f"{name}  (N={distribution.diameter.size})",
                )
            self._axis.legend(fontsize="small")
        self._axis.set_xlabel("Diameter (nm)")
        self._axis.set_ylabel("Weighted density (1/nm)")
        self._axis.set_title("Common diameter histogram")
        self.figure.tight_layout()
        self.draw_idle()


class ECDFCanvas(FigureCanvasQTAgg):
    """Bin-independent weighted ECDF comparison."""

    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(6.4, 4.4), dpi=100)
        super().__init__(self.figure)
        self.setParent(parent)
        self._axis = self.figure.add_subplot(111)

    def set_series(self, series: list[tuple[str, DiameterDistribution]]) -> None:
        self._axis.clear()
        self._axis.grid(alpha=0.2)
        if not series:
            self._axis.text(
                0.5,
                0.5,
                "No distributions available",
                ha="center",
                va="center",
                transform=self._axis.transAxes,
            )
            self._axis.set_title("Diameter ECDF")
            self.draw_idle()
            return
        for name, distribution in series:
            if not distribution.diameter.size:
                continue
            order = np.argsort(distribution.diameter, kind="stable")
            x = distribution.diameter[order]
            y = np.cumsum(distribution.weight[order]) / distribution.weight.sum()
            self._axis.step(
                x * 1000.0,
                y,
                where="post",
                label=f"{name}  (N={distribution.diameter.size})",
                color=SERIES_COLORS.get(name),
            )
        self._axis.legend(fontsize="small")
        self._axis.set_xlabel("Diameter (nm)")
        self._axis.set_ylabel("Cumulative weight")
        self._axis.set_title("Diameter ECDF")
        self.figure.tight_layout()
        self.draw_idle()


def distribution_quantile_table(
    series: list[tuple[str, DiameterDistribution]],
) -> list[tuple[str, ...]]:
    """Rows for the distribution summary table: series, N, mean, median, IQR, P05, P95."""
    rows: list[tuple[str, ...]] = []
    for name, distribution in series:
        if not distribution.diameter.size:
            rows.append((name, "0", "—", "—", "—", "—", "—"))
            continue
        summary = summarize_distribution(distribution)
        iqr = (
            f"{summary.p25 * 1000.0:.5g}–{summary.p75 * 1000.0:.5g}"
            if summary.p25 is not None and summary.p75 is not None
            else "—"
        )
        rows.append(
            (
                name,
                str(summary.n),
                _fmt(summary.weighted_mean * 1000.0),
                _fmt(summary.weighted_median * 1000.0),
                iqr,
                _fmt(summary.p05 * 1000.0),
                _fmt(summary.p95 * 1000.0),
            )
        )
    return rows


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.5g}"
