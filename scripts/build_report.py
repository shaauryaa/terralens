"""Build reports/progress_report.pdf (A4, exactly 2 pages) from reports/progress_report.md.

The Markdown file is the single source of the report text; this script only lays it out with reportlab.
Supported Markdown (a small subset):
    # Title                      paragraphs between it and the first ## heading are subtitle lines
    ## Heading                   section heading (navy)
    plain lines                  paragraphs (lines are joined until a blank line)
    - item                       bullet list
    | a | b |                    pipe table (first row = header, `---:` = right-aligned column)
    ![alt](path "w=3.9")         figure; several on one line are placed side by side (w = width in inches)
    *Figure ...*                 caption (a paragraph that starts with "*Figure")
    ---                          rule; everything after it is set as the small footer text
Inline: **bold**, *italic*, `code`, [text](url).

The page count is checked with pypdf. If the report is longer than 2 pages, text and figures are scaled down in
small steps until it fits; the script fails if it cannot reach exactly 2 pages.

Usage:
    python scripts/build_report.py
    python scripts/build_report.py --preview outputs/report_preview    # also render each page to PNG (PyMuPDF)
    python scripts/build_report.py --limit 1                           # dry run: parse and list blocks only
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "reports" / "progress_report.md"
PDF = ROOT / "reports" / "progress_report.pdf"
TARGET_PAGES = 2
SEED = 42  # nothing is sampled here; kept for the script convention

# Plain black text, headings, table headers and rules; colour only inside the figures.
INK, RULE = "#000000", "#000000"
FONT_FILES = {"": "calibri.ttf", "-Bold": "calibrib.ttf", "-Italic": "calibrii.ttf", "-BoldItalic": "calibriz.ttf"}


# ---------------------------------------------------------------- Markdown parsing (pure, tested)
IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"w=([\d.]+)")?\)')


def inline(text: str) -> str:
    """Markdown inline markup -> reportlab paragraph markup (escapes &, <, > first)."""
    t = html.escape(text, quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2"><u>\1</u></link>', t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="-0.5">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<i>\1</i>", t)
    return t


def parse_table(lines: list[str]) -> tuple[list[list[str]], list[str]]:
    rows, aligns = [], []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            aligns = ["RIGHT" if c.endswith(":") else "LEFT" for c in cells]
            continue
        rows.append(cells)
    return rows, aligns or ["LEFT"] * len(rows[0])


def parse(md: str) -> list[tuple]:
    """Blocks: (title, s) (subtitle, s) (h2, s) (p, s) (caption, s) (bullets, [s]) (table, rows, aligns)
    (images, [(path, width)]) (rule,) (footer, s)."""
    blocks: list[tuple] = []
    lines = md.splitlines()
    i, after_rule, para = 0, False, []

    def flush():
        if para:
            text = " ".join(s.strip() for s in para)
            if any(b[0] == "title" for b in blocks) and not any(b[0] == "h2" for b in blocks):
                blocks.append(("subtitle", text.strip("*")))
            elif after_rule:
                blocks.append(("footer", text))
            elif text.startswith("*Figure"):
                blocks.append(("caption", text.strip("*")))
            else:
                blocks.append(("p", text))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("<!--"):  # HTML comment, possibly over several lines: not part of the report
            flush()
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
        elif not s:
            flush()
        elif s.startswith("# "):
            flush()
            blocks.append(("title", s[2:].strip()))
        elif s.startswith("## "):
            flush()
            blocks.append(("h2", s[3:].strip()))
        elif s == "---":
            flush()
            blocks.append(("rule",))
            after_rule = True
        elif s.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():  # continuation line
                    items[-1] += " " + lines[i].strip()
                    i += 1
            blocks.append(("bullets", items))
            continue
        elif s.startswith("|"):
            flush()
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            blocks.append(("table", *parse_table(tbl)))
            continue
        elif IMG_RE.fullmatch(s) or (s.startswith("![") and IMG_RE.sub("", s).strip() == ""):
            flush()
            blocks.append(("images", [(m.group(2), float(m.group(3) or 7.0)) for m in IMG_RE.finditer(s)]))
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


# ---------------------------------------------------------------- PDF layout
def register_fonts() -> str:
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_dir = Path("C:/Windows/Fonts")
    if not all((font_dir / f).exists() for f in FONT_FILES.values()):
        return "Helvetica"  # fallback: built-in font (some symbols may not render)
    for suffix, fname in FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(f"Calibri{suffix}", str(font_dir / fname)))
    addMapping("Calibri", 0, 0, "Calibri")
    addMapping("Calibri", 1, 0, "Calibri-Bold")
    addMapping("Calibri", 0, 1, "Calibri-Italic")
    addMapping("Calibri", 1, 1, "Calibri-BoldItalic")
    return "Calibri"


def build_pdf(blocks: list[tuple], out: Path, scale: float, base_dir: Path) -> int:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (HRFlowable, Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    font = register_fonts()
    bold = f"{font}-Bold" if font == "Calibri" else "Helvetica-Bold"
    fs = 10.0 * scale
    body = ParagraphStyle("body", fontName=font, fontSize=fs, leading=fs * 1.2, textColor=INK, spaceAfter=2.2 * scale)
    styles = {
        "title": ParagraphStyle("title", parent=body, fontName=bold, fontSize=15 * scale, leading=17.5 * scale,
                                alignment=TA_CENTER, spaceAfter=2 * scale),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=9.6 * scale, leading=11.2 * scale,
                                   alignment=TA_CENTER, spaceAfter=0.5 * scale),
        "h2": ParagraphStyle("h2", parent=body, fontName=bold, fontSize=11.5 * scale, leading=13.5 * scale,
                             spaceBefore=5 * scale, spaceAfter=1.8 * scale, keepWithNext=1),
        "p": body,
        "caption": ParagraphStyle("caption", parent=body, fontSize=8.8 * scale, leading=10.2 * scale,
                                  alignment=TA_CENTER, spaceAfter=3 * scale),
        "footer": ParagraphStyle("footer", parent=body, fontSize=8.4 * scale, leading=9.8 * scale),
        "cell": ParagraphStyle("cell", parent=body, fontSize=9.2 * scale, leading=10.6 * scale, spaceAfter=0),
    }
    cell_r = ParagraphStyle("cellr", parent=styles["cell"], alignment=2)
    head = ParagraphStyle("head", parent=styles["cell"], fontName=bold)
    head_r = ParagraphStyle("headr", parent=head, alignment=2)

    margin = 0.5 * inch
    width = A4[0] - 2 * margin
    story = []
    for k, b in enumerate(blocks):
        kind = b[0]
        if kind == "caption" and k > 0 and blocks[k - 1][0] == "images":
            continue  # already placed together with its figure
        if kind == "title":  # one line: shrink the title font until it fits the text width
            from reportlab.pdfbase.pdfmetrics import stringWidth

            st = styles["title"]
            while stringWidth(b[1], bold, st.fontSize) > width - 4 and st.fontSize > 10:
                st = ParagraphStyle("title", parent=st, fontSize=st.fontSize - 0.25, leading=st.fontSize * 1.15)
            story.append(Paragraph(inline(b[1]), st))
        elif kind in ("subtitle", "h2", "p", "caption", "footer"):
            story.append(Paragraph(inline(b[1]), styles[kind]))
        elif kind == "bullets":
            items = [ListItem(Paragraph(inline(t), body), leftIndent=9 * scale, value="•") for t in b[1]]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=9 * scale,
                                      bulletFontSize=fs, bulletColor=INK, spaceAfter=2 * scale))
        elif kind == "table":
            rows, aligns = b[1], b[2]
            data = [[Paragraph(inline(c), (head_r if a == "RIGHT" else head) if r == 0 else
                               (cell_r if a == "RIGHT" else styles["cell"])) for c, a in zip(row, aligns)]
                    for r, row in enumerate(rows)]
            ncol = len(rows[0])
            widths = _col_widths(rows, width, ncol, styles["cell"].fontSize)
            t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
            black = colors.HexColor(RULE)
            style = [("LINEABOVE", (0, 0), (-1, 0), 0.9, black), ("LINEBELOW", (0, 0), (-1, 0), 0.6, black),
                     ("LINEBELOW", (0, -1), (-1, -1), 0.9, black),
                     ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                     ("TOPPADDING", (0, 0), (-1, -1), 1.2 * scale), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * scale),
                     ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
            t.setStyle(TableStyle(style))
            story += [KeepTogether([t]), Spacer(1, 3 * scale)]  # never split a table across pages
        elif kind == "images":
            cells = []
            for path, w in b[1]:
                p = (base_dir / path).resolve()
                img = Image(str(p))
                ratio = img.imageHeight / img.imageWidth
                w_pt = min(w * inch * scale, width)
                cells.append(Image(str(p), width=w_pt, height=w_pt * ratio))
            if len(cells) == 1:
                cells[0].hAlign = "CENTER"
                fig = cells[0]
            else:
                fig = Table([cells], colWidths=[c.drawWidth + 4 for c in cells], hAlign="CENTER")
                fig.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                         ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                         ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
            group = [fig, Spacer(1, 1.5 * scale)]
            if k + 1 < len(blocks) and blocks[k + 1][0] == "caption":  # keep a figure and its caption together
                group.append(Paragraph(inline(blocks[k + 1][1]), styles["caption"]))
            story.append(KeepTogether(group))
        elif kind == "rule" and k > 0 and blocks[k - 1][0] == "table":
            story.append(Spacer(1, 4 * scale))  # the table's own bottom rule already separates the footer
        elif kind == "rule":
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(RULE), spaceBefore=3 * scale,
                                    spaceAfter=2 * scale))

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=0.42 * inch,
                            bottomMargin=0.42 * inch, title="TerraLens Round 2 progress report",
                            author="Shaurya Nigam, Nakshatra Bhateja")
    doc.build(story)  # no running header or footer
    return page_count(out)


def _col_widths(rows: list[list[str]], width: float, ncol: int, font_size: float = 9.0) -> list[float]:
    """Column widths that fill ``width``. Short columns get their natural width (longest cell, so they never wrap);
    the space left is shared by the long columns in proportion to their natural width ("water filling")."""
    plain = [[re.sub(r"[*`]", "", c) for c in row] for row in rows]
    char_w = 0.5 * font_size  # average Calibri character width, in points
    natural = [(max(len(r[j]) for r in plain) + 2) * char_w for j in range(ncol)]
    if sum(natural) <= width:
        return [width * n / sum(natural) for n in natural]
    widths = [0.0] * ncol
    left, todo = width, sorted(range(ncol), key=lambda j: natural[j])
    while todo:
        share = left / len(todo)
        j = todo[0]
        if natural[j] <= share:
            widths[j] = natural[j]
            left -= natural[j]
            todo.pop(0)
        else:
            total = sum(natural[i] for i in todo)
            for i in todo:
                widths[i] = left * natural[i] / total
            break
    return widths


def page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def fit(blocks: list[tuple], out: Path, base_dir: Path) -> tuple[float, int]:
    """Largest scale (1.00, 0.99, ... 0.80) at which the PDF has exactly TARGET_PAGES pages."""
    scale = 1.0
    while scale >= 0.8 - 1e-9:
        n = build_pdf(blocks, out, scale, base_dir)
        if n == TARGET_PAGES:
            return scale, n
        if n < TARGET_PAGES:
            return scale, n
        scale = round(scale - 0.01, 2)
    return scale, n


def render_preview(pdf: Path, out_dir: Path, dpi: int = 110) -> list[Path]:
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(str(pdf)) as doc:
        for i, page in enumerate(doc, start=1):
            p = out_dir / f"progress_report_page{i}.png"
            page.get_pixmap(dpi=dpi).save(str(p))
            paths.append(p)
    return paths


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md", type=Path, default=MD)
    p.add_argument("--out", type=Path, default=PDF)
    p.add_argument("--preview", type=Path, default=None, help="folder for PNG previews of each page")
    p.add_argument("--limit", type=int, default=None, help="dry run: parse and list the first N blocks")
    args = p.parse_args(argv)
    blocks = parse(args.md.read_text(encoding="utf-8"))
    if args.limit:
        for b in blocks[: args.limit]:
            print(b[0], str(b[1:])[:100])
        print(f"{len(blocks)} blocks; dry run, no PDF written")
        return
    scale, n = fit(blocks, args.out, args.md.parent)
    print(f"wrote {args.out} ({n} pages, scale {scale:.2f})")
    if args.preview:
        for path in render_preview(args.out, args.preview):
            print("preview", path)
    if n != TARGET_PAGES:
        sys.exit(f"page count is {n}, not {TARGET_PAGES}: shorten the text in {args.md.name}")


if __name__ == "__main__":
    main()
