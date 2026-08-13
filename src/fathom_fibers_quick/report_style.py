"""Shared scientific-report visual design system (HTML/CSS helpers).

Screen + print CSS, semantic table/card/details helpers, short method names
and plain-language flag definitions.  Pure reporting-layer presentation; no
scientific computation happens here.
"""

from __future__ import annotations

import html as html_module
from typing import Any

CSS = """
:root {
  --bg: #faf8f4;
  --surface: #ffffff;
  --surface-soft: #f4f1ea;
  --text: #23262b;
  --muted: #6a6f78;
  --dim: #9aa0a8;
  --border: #e2ddd2;
  --border-strong: #c9c2b4;
  --accent: #b37d1f;
  --accent-soft: #f6e9d0;
  --info: #3f7f93;
  --info-soft: #e4f0f4;
  --success: #2f9e63;
  --success-soft: #e3f3ea;
  --warning: #b7791f;
  --warning-soft: #fff4dd;
  --error: #b3403f;
  --error-soft: #f9e7e6;
}
* { box-sizing: border-box; }
html { font-size: 15px; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--text);
  background: var(--bg);
  margin: 0;
  line-height: 1.5;
}
.wrap { max-width: 1320px; margin: 0 auto; padding: 24px 28px 64px; }
header.report-head { margin-bottom: 26px; }
h1 { font-size: 1.7rem; margin: 0 0 2px; letter-spacing: 0.2px; }
.lead { color: var(--muted); margin: 0 0 14px; font-size: 1.02rem; }
h2 {
  font-size: 1.2rem;
  margin: 34px 0 10px;
  padding-bottom: 6px;
  border-bottom: 2px solid var(--border-strong);
  color: #1d1f23;
}
h3 { font-size: 1.02rem; margin: 22px 0 8px; }
p { margin: 8px 0; }
code, pre {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.86rem;
  background: var(--surface-soft);
  border-radius: 4px;
  padding: 1px 5px;
}
pre { padding: 10px 12px; overflow-x: auto; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 16px;
  font-size: 0.92rem;
}
th, td {
  border: 1px solid var(--border);
  padding: 5px 9px;
  text-align: right;
  vertical-align: top;
}
th { background: var(--surface-soft); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tr:nth-child(even) td { background: var(--surface); }
tr:nth-child(odd) td { background: #fcfbf8; }
table.grouped th.group { background: var(--accent-soft); color: #5c440f; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 12px 0; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.card .value { font-size: 1.35rem; font-weight: 650; color: #1d1f23; }
.card .label { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
.card .sub { font-size: 0.78rem; color: var(--dim); margin-top: 2px; }
.badge {
  display: inline-block;
  border: 1px solid var(--border-strong);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.78rem;
  font-weight: 600;
  margin: 0 6px 6px 0;
  color: var(--text);
}
.badge.ok { background: var(--success-soft); border-color: #b9dcc9; }
.badge.warn { background: var(--warning-soft); border-color: #e4cfa0; }
.badge.info { background: var(--info-soft); border-color: #bcd6e0; }
.badge.exp { background: var(--accent-soft); border-color: #d9c187; }
.meta-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.chip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 0.84rem;
}
.chip b { font-weight: 600; }
figure { margin: 14px 0 20px; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }
figcaption { font-size: 0.82rem; color: var(--muted); margin-top: 6px; }
.note {
  border-left: 4px solid var(--warning);
  background: var(--warning-soft);
  padding: 10px 14px;
  border-radius: 0 6px 6px 0;
  margin: 12px 0;
  font-size: 0.92rem;
}
.info {
  border-left: 4px solid var(--info);
  background: var(--info-soft);
  padding: 10px 14px;
  border-radius: 0 6px 6px 0;
  margin: 12px 0;
  font-size: 0.92rem;
}
details {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin: 10px 0;
  padding: 8px 12px;
  background: var(--surface);
}
summary { cursor: pointer; font-weight: 600; color: #33383f; }
.detail-scroll { max-height: 480px; overflow: auto; }
.toc { columns: 2; column-gap: 30px; margin: 12px 0; }
.toc ol { margin: 0; padding-left: 20px; }
.toc li { margin: 3px 0; }
a { color: var(--info); text-decoration: none; }
a:hover { text-decoration: underline; }
.image-nav { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 20px; }
.image-nav a {
  border: 1px solid var(--border-strong);
  border-radius: 5px;
  padding: 3px 9px;
  font-size: 0.84rem;
  color: var(--text);
  background: var(--surface);
}
.image-nav a:hover { border-color: var(--accent); }
.status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.flag-list { margin: 4px 0; }
@media (max-width: 900px) {
  .toc { columns: 1; }
  .wrap { padding: 16px 14px 40px; }
}
@media print {
  body { background: #ffffff; }
  .wrap { max-width: none; padding: 0; }
  .toc, .image-nav { display: none; }
  details { border: none; padding: 0; }
  details summary { display: none; }
  details[open] > * { display: inherit; }
  figure { break-inside: avoid; }
  h2 { break-after: avoid; page-break-after: avoid; }
  table { font-size: 0.8rem; }
  a { color: var(--text); text-decoration: none; }
  .badge, .card, .chip { border-color: #d5d5d5; }
}
"""


def page(title: str, *, head_html: str = "", body_html: str = "") -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_module.escape(title)}</title>
<style>{CSS}</style>
</head><body><div class="wrap">
{head_html}
{body_html}
</div></body></html>"""


def report_header(title: str, lead: str, chips: list[tuple[str, str]], badges: list[tuple[str, str]]) -> str:
    chips_html = "".join(
        f"<span class='chip'><b>{html_module.escape(label)}</b> {html_module.escape(value)}</span>"
        for label, value in chips
    )
    badges_html = "".join(
        f"<span class='badge {cls}'>{html_module.escape(text)}</span>" for text, cls in badges
    )
    return f"""<header class="report-head">
<h1>{html_module.escape(title)}</h1>
<p class="lead">{html_module.escape(lead)}</p>
<div class="meta-chips">{chips_html}</div>
<div>{badges_html}</div>
</header>"""


def section(heading: str, body: str, *, id_: str | None = None) -> str:
    anchor = f" id='{id_}'" if id_ else ""
    return f"<section{anchor}><h2>{html_module.escape(heading)}</h2>{body}</section>"


def subsection(heading: str, body: str) -> str:
    return f"<h3>{html_module.escape(heading)}</h3>{body}"


def table(headers: list[str], rows: list[list[Any]], *, classes: str = "") -> str:
    head = "".join(f"<th>{html_module.escape(str(h))}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f"<td>{value if isinstance(value, str) else html_module.escape(str(value))}</td>"
            for value in row
        )
        body += f"<tr>{cells}</tr>"
    return f"<table class='{classes}'><tr>{head}</tr>{body}</table>"


def grouped_table(groups: list[tuple[str, list[str], list[list[Any]]]]) -> str:
    """Table with group header rows (name spans all columns)."""
    body = ""
    for group_name, _headers, rows in groups:
        body += f"<tr class='group'><th class='group' colspan='99'>{html_module.escape(group_name)}</th></tr>"
        for row in rows:
            cells = "".join(
                f"<td>{value if isinstance(value, str) else html_module.escape(str(value))}</td>"
                for value in row
            )
            body += f"<tr>{cells}</tr>"
    return f"<table class='grouped'>{body}</table>"


def cards(items: list[tuple[str, str, str]]) -> str:
    """(value, label, sub) triples rendered as a responsive card grid."""
    inner = "".join(
        f"<div class='card'><div class='value'>{html_module.escape(value)}</div>"
        f"<div class='label'>{html_module.escape(label)}</div>"
        f"<div class='sub'>{html_module.escape(sub)}</div></div>"
        for value, label, sub in items
    )
    return f"<div class='cards'>{inner}</div>"


def figure(src: str, alt: str, caption: str, *, width: int | None = None) -> str:
    style = f" style='max-width:{width}px'" if width else ""
    return (
        f"<figure><img src='{html_module.escape(src)}' alt='{html_module.escape(alt)}'{style}>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def details(summary: str, body: str) -> str:
    return f"<details><summary>{html_module.escape(summary)}</summary>{body}</details>"


def note(text: str) -> str:
    return f"<div class='note'>{text}</div>"


def info(text: str) -> str:
    return f"<div class='info'>{text}</div>"


def flag_definition(flag: str) -> str:
    return FLAG_DEFINITIONS.get(flag, "")


FLAG_DEFINITIONS = {
    "POSSIBLE_CROSSING": (
        "Local geometry may contain a projected fiber crossing; the refinement abstains."
    ),
    "AMBIGUOUS_LOCAL_WIDTH": (
        "One or more local width diagnostics are ambiguous (coherence, asymmetry, "
        "possible crossing, or boundary normal mismatch); the sample is rejected."
    ),
    "LOW_ORIENTATION_COHERENCE": "Local fiber direction is poorly determined.",
    "MISSING_POSITIVE_EDGE": "A valid boundary was not found on one side of the fiber.",
    "MISSING_NEGATIVE_EDGE": "A valid boundary was not found on the other side of the fiber.",
    "HIGH_ASYMMETRY": "The two boundary distances are strongly unequal.",
    "EDGE_TANGENT_MISMATCH": "The local boundary tangent disagrees with the centerline direction.",
    "EDGE_NORMAL_MISMATCH": "The boundary's inward normal disagrees with the sampled normal.",
    "PROFILE_AMBIGUOUS_EDGE": "Several comparable intensity-gradient candidates exist.",
    "PROFILE_EDGE_MINUS_NOT_FOUND": "No usable intensity gradient was found on one side.",
    "PROFILE_EDGE_PLUS_NOT_FOUND": "No usable intensity gradient was found on the other side.",
    "PROFILE_NONPOSITIVE_WIDTH": "The refined profile produced a non-positive width.",
    "PROFILE_REJECTED_FROM_PAIRED_EDGE": "The paired-edge prior itself was rejected, so the profile abstained.",
    "REFINEMENT_SEGMENT_TOO_SHORT": "Too few supported samples exist for a stable local refined segment.",
    "REFINED_ORIENTATION_DISAGREEMENT": (
        "The refined centerline tangent disagrees strongly with the orientation field."
    ),
    "LIKELY_MERGED": "The local geometry suggests two nearby fibers may be merged.",
    "WIDTH_TOO_VARIABLE": "Local width varies strongly along the candidate.",
    "LOW_ELONGATION": "The candidate is not sufficiently elongated.",
    "TOUCHES_ROI_EDGE": "The candidate touches the analysis region boundary.",
}

SHORT_NAMES = {
    "MATLAB SIMPoly": "MATLAB SIMPoly",
    "Python SIMPoly": "SIMPoly Python",
    "Fathom Local": "Local",
    "Fathom Field (EDT)": "Raw EDT",
    "Field Paired Edge": "Raw Edge",
    "Field Intensity Profile": "Raw Profile",
    "Ribbon Refined EDT": "Ribbon EDT",
    "Ribbon Refined Edge": "Ribbon Edge",
    "Ribbon Refined Profile": "Ribbon Profile",
    "Manual 5×5": "Manual",
    "Consensus": "Consensus",
}


def short_name(name: str) -> str:
    return SHORT_NAMES.get(name, name)
