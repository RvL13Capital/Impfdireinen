"""Render RESEARCH.md into a polished, print-ready PDF (docs/Quiet-Volume-Research.pdf).

Markdown → styled HTML → PDF via WeasyPrint, with the committed figures embedded,
an academic stylesheet, running headers and page numbers. WeasyPrint also emits a
navigable PDF outline from the headings automatically.

    pip install markdown weasyprint
    python docs/build_research_pdf.py
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "RESEARCH.md"
OUT = ROOT / "docs" / "Quiet-Volume-Research.pdf"

CSS = """
@page {
  size: A4; margin: 20mm 18mm 22mm 18mm;
  @top-left  { content: "Quiet-Volume (vpts) · Validation Log"; font: 8pt 'Helvetica'; color: #8a93a3; }
  @top-right { content: "v1.10.0"; font: 8pt 'Helvetica'; color: #8a93a3; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt 'Helvetica'; color: #8a93a3; }
  @bottom-left  { content: "Not financial advice · research & education"; font: 7.5pt 'Helvetica'; color: #b3bac6; }
}
@page :first { @top-left { content: ""; } @top-right { content: ""; } }

html { font-size: 10.5pt; }
body { font-family: Georgia, 'Times New Roman', serif; color: #1c2330; line-height: 1.5; }

/* Cover banner (injected before the body) */
.cover { border-bottom: 3px solid #1a5f9e; padding-bottom: 10px; margin-bottom: 18px; }
.cover .kicker { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; letter-spacing: .14em;
  text-transform: uppercase; color: #1a5f9e; font-weight: 700; }
.cover .meta { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; color: #6b7382; margin-top: 4px; }

h1, h2, h3, h4 { font-family: Helvetica, Arial, sans-serif; color: #11203a; line-height: 1.25; }
h1 { font-size: 21pt; margin: 2px 0 6px; }
h2 { font-size: 14.5pt; margin: 20px 0 6px; padding-bottom: 4px; border-bottom: 1px solid #d7deea;
  color: #1a5f9e; }
h3 { font-size: 11.5pt; margin: 14px 0 4px; color: #243a5e; }
p { margin: 6px 0; text-align: justify; }
a { color: #1a5f9e; text-decoration: none; }
strong { color: #11203a; }

blockquote { margin: 12px 0; padding: 10px 14px; background: #eef4fb; border-left: 4px solid #1a5f9e;
  border-radius: 0 4px 4px 0; }
blockquote p { margin: 4px 0; }

table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 8.6pt;
  font-family: Helvetica, Arial, sans-serif; }
th, td { border: 1px solid #cfd6e2; padding: 4px 7px; text-align: left; vertical-align: top; }
thead th { background: #1a5f9e; color: #fff; border-color: #1a5f9e; }
tbody tr:nth-child(even) { background: #f3f6fb; }

code { font-family: 'DejaVu Sans Mono', Menlo, monospace; font-size: 8.6pt;
  background: #eef1f6; padding: 1px 4px; border-radius: 3px; color: #243a5e; }
pre { background: #0f1620; color: #d6deeb; border-radius: 6px; padding: 10px 12px; margin: 10px 0;
  font-size: 8pt; line-height: 1.35; overflow: hidden; white-space: pre-wrap; }
pre code { background: transparent; color: inherit; padding: 0; }

img { max-width: 84%; display: block; margin: 12px auto; border: 1px solid #2a3340; border-radius: 6px; }
p[align="center"] { text-align: center; }

hr { border: none; border-top: 1px solid #d7deea; margin: 18px 0; }
h2, h3 { break-after: avoid; }
table, blockquote, pre, img { break-inside: avoid; }
"""

COVER = (
    '<div class="cover">'
    '<div class="kicker">Quiet-Volume · vpts — Quantitative Validation Study</div>'
    f'<div class="meta">Thirteen experiments · purged CPCV + permutation + survivorship · '
    f'compiled {_dt.date.today().isoformat()}</div>'
    "</div>"
)


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    # Drop the badge row and any HTML comments that don't belong in a print document.
    text = re.sub(r"^\s*!\[.*?\]\(https://img\.shields\.io.*?\)\s*$", "", text, flags=re.M)

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{COVER}{body}</body></html>")

    HTML(string=html, base_url=str(ROOT) + "/").write_pdf(str(OUT))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
