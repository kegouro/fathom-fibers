"""In-app help: Quick Start, User Guide, Methods and Keyboard Shortcuts.

One reusable themed dialog; no separate giant dialogs per topic.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
)

from .widgets.workspace_panels import method_entries

PAGE_QUICK_START = 0
PAGE_USER_GUIDE = 1
PAGE_METHODS = 2
PAGE_SHORTCUTS = 3

QUICK_START_HTML = """
<h2>Quick Start</h2>
<p><b>OPEN</b><br>
Click <b>Open Dataset</b> and choose the folder with your SEM images.</p>
<p><b>ANALYZE</b> (<code>A</code>)<br>
If the header says analysis is incomplete, click <b>Analyze Dataset</b>
(equivalent to <i>Run missing</i>). Then inspect the bottom tabs:
<b>Distribution</b>, <b>Comparison</b>, <b>Quality</b>. Use <b>Layers</b>
to overlay centerlines and edges on the image.</p>
<p><b>MEASURE</b> (<code>M</code>)<br>
If a human reference is required, complete the <b>Manual 5×5</b> grid:
draw a perpendicular width line on each target; measurements autosave.</p>
<p><b>REPORT</b> (<code>R</code>)<br>
Click <b>Generate Dataset Scientific Report</b>, then
<b>Export Analysis Bundle</b> to save results and figures to a folder.</p>
"""

USER_GUIDE_HTML = """
<h2>User Guide</h2>
<p><b>Analyze</b> — the default workspace: dataset navigator on the left, SEM
image in the center, image summary on the right, and three bottom tabs
(Distribution, Comparison, Quality). The header action is
<b>Explore Results</b> when everything is already analyzed, or
<b>Analyze Dataset</b> when some images still need analysis.</p>
<p><b>Three run actions.</b> <i>Run Methods</i> analyzes the current image.
<i>Run missing</i> analyzes only images without a valid cached result
(preferred for normal continuation). <i>Run all dataset</i> recomputes
everything — use intentionally. Results are cached per image; switching
images or modes never reruns anything.</p>
<p><b>Manual 5×5</b> — a sparse human reference: 25 targets per image
(400 for a 16-image dataset). Draw a perpendicular width line on each
target; measurements autosave immediately. <i>Enter</i> next, <i>Backspace</i>
previous, <i>Delete</i> remove, <i>Esc</i> cancel.</p>
<p><b>Report</b> — generate the dataset scientific report (HTML) and export
the analysis bundle (CSV/JSON figures + report + provenance).</p>
<p><b>Advanced</b> — technical diagnostics (Methods, Measurements, History,
Analysis, Batch Review) for expert use.</p>
<p>See <code>docs/USER_GUIDE.md</code> for the complete guide.</p>
"""

METHODS_HTML_TEMPLATE = """
<h2>Methods guide</h2>
<table>
<tr><th>Method</th><th>Purpose</th><th>Status</th></tr>
{rows}
</table>
<p>Click a row in the in-app dialog for full scientific caveats.
<b>Oriented Ribbon V1 is EXPERIMENTAL</b>: known-truth synthetic geometry
validates the centerline mechanism; real SEM comparisons show method
behavior/agreement, not known absolute accuracy.</p>
"""

SHORTCUTS_HTML = """
<h2>Keyboard shortcuts</h2>
<table>
<tr><th>Key</th><th>Action</th></tr>
<tr><td>A</td><td>Analyze mode</td></tr>
<tr><td>M</td><td>Manual 5×5 mode (also selects the width tool)</td></tr>
<tr><td>R</td><td>Report mode</td></tr>
<tr><td>Left / Right</td><td>Previous / next image</td></tr>
<tr><td>F</td><td>Fit image</td></tr>
<tr><td>1</td><td>1:1 pixels</td></tr>
<tr><td>0</td><td>Reset view</td></tr>
<tr><td>Ctrl+O</td><td>Open image</td></tr>
<tr><td>Ctrl+S / Ctrl+Shift+S</td><td>Save / Save project as</td></tr>
<tr><td>Ctrl+E</td><td>Export CSV</td></tr>
<tr><td>Ctrl+R</td><td>Generate current image report</td></tr>
<tr><td>Ctrl+Z / Ctrl+Shift+Z</td><td>Undo / Redo</td></tr>
<tr><td>Delete</td><td>Delete selected measurement</td></tr>
<tr><td colspan='2'><b>Manual 5×5:</b> Enter accept/next · Backspace previous · Delete remove · Esc cancel</td></tr>
</table>
"""


class HelpDialog(QDialog):
    """Reusable themed help dialog with four pages."""

    def __init__(self, parent=None, *, page: int = PAGE_QUICK_START) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fathom Fibers — Help")
        self.setMinimumSize(720, 520)
        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        self.pages = QListWidget()
        self.pages.addItems(["Quick Start", "User Guide", "Methods Guide", "Keyboard Shortcuts"])
        self.pages.setFixedWidth(150)
        layout.addWidget(self.pages)

        self.stack = QStackedWidget()
        self.quick_start = self._browser(QUICK_START_HTML, footer=True)
        self.user_guide = self._browser(USER_GUIDE_HTML)
        self.methods = self._browser(_methods_html())
        self.shortcuts = self._browser(SHORTCUTS_HTML)
        for widget in (self.quick_start, self.user_guide, self.methods, self.shortcuts):
            self.stack.addWidget(widget)
        layout.addWidget(self.stack, 1)

        self.dont_show_check = QCheckBox("Don't show automatically again")
        self.dont_show_check.setVisible(False)
        footer = QHBoxLayout()
        footer.addWidget(self.dont_show_check)
        footer.addStretch(1)
        outer = QVBoxLayout(self)
        outer.addLayout(layout, 1)
        outer.addLayout(footer)
        self.pages.currentRowChanged.connect(self._switch)
        self.pages.setCurrentRow(page)

    def _browser(self, html: str, *, footer: bool = False) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setHtml(html)
        browser.setOpenExternalLinks(True)
        return browser

    def _switch(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        self.dont_show_check.setVisible(row == PAGE_QUICK_START)


def _methods_html() -> str:
    rows = ""
    for name, purpose, status, _details in method_entries():
        badge = status
        rows += f"<tr><td><b>{name}</b></td><td>{purpose}</td><td>{badge}</td></tr>"
    return METHODS_HTML_TEMPLATE.format(rows=rows)


def show_help(parent=None, *, page: int = PAGE_QUICK_START, quick_start_seen: bool = True) -> HelpDialog:
    """Open the help dialog; first-run Quick Start shows the opt-out checkbox."""
    dialog = HelpDialog(parent, page=page)
    if page == PAGE_QUICK_START and not quick_start_seen:
        dialog.dont_show_check.setChecked(False)
    return dialog
