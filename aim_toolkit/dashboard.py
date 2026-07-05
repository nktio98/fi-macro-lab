"""
Self-contained HTML dashboard generator.

Produces a single .html file with all charts (base64-embedded PNGs) and
tables -- shareable by email, no server, no dependencies. Swap-in point
for a Streamlit/Power BI front end later; the section API stays the same.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

_CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;margin:0;background:#f4f6f8;color:#1a2733}
header{background:#003781;color:#fff;padding:22px 40px}
header h1{margin:0;font-size:22px} header p{margin:4px 0 0;opacity:.8;font-size:13px}
.section{background:#fff;margin:22px 40px;padding:20px 26px;border-radius:8px;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}
.section h2{margin:0 0 6px;font-size:17px;color:#003781}
.section p.note{margin:2px 0 12px;font-size:13px;color:#546575}
img{max-width:100%;border:1px solid #e2e7ec;border-radius:4px}
table{border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{border:1px solid #dde3e9;padding:5px 10px;text-align:right}
th{background:#eef2f6} td:first-child,th:first-child{text-align:left}
footer{padding:14px 40px;font-size:12px;color:#8a97a5}
"""


class Dashboard:
    def __init__(self, title: str, subtitle: str = ""):
        self.title, self.subtitle = title, subtitle
        self.sections: list[str] = []

    def add_section(self, heading: str, note: str = "",
                    image_path: str | None = None,
                    table: pd.DataFrame | None = None):
        parts = [f"<h2>{heading}</h2>"]
        if note:
            parts.append(f"<p class='note'>{note}</p>")
        if table is not None:
            parts.append(table.to_html(border=0))
        if image_path and Path(image_path).exists():
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            parts.append(f"<img src='data:image/png;base64,{b64}'/>")
        self.sections.append("<div class='section'>" + "".join(parts) + "</div>")
        return self

    def save(self, path: str, footer: str = ""):
        html = (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>{self.title}</title><style>{_CSS}</style></head><body>"
                f"<header><h1>{self.title}</h1><p>{self.subtitle}</p></header>"
                + "".join(self.sections)
                + f"<footer>{footer}</footer></body></html>")
        Path(path).write_text(html, encoding="utf-8")
        return path
