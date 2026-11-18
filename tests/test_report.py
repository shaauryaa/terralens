"""Tests for scripts/make_report_figures.py and scripts/build_report.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mrf = _load("make_report_figures")
br = _load("build_report")

DS = {
    "class_by_lang": {"hi": {"Favour": 6, "Against": 1, "None": 13}, "bn": {"Favour": 2, "Against": 1, "None": 7}},
    "class_by_split": {"train": {"Favour": 5, "Against": 1, "None": 14}, "val": {"Favour": 2, "Against": 0, "None": 3},
                       "test": {"Favour": 1, "Against": 1, "None": 3}},
    "script_by_lang": {"hi": {"bengali": 0, "devanagari": 9, "latin": 10, "mixed": 1},
                       "bn": {"bengali": 7, "devanagari": 0, "latin": 2, "mixed": 1}},
}


def test_script_mix_counts_native_script_per_language():
    assert mrf.script_mix_totals(DS) == {"native": 16, "romanised": 12, "mixed": 2}


def test_pipeline_numbers_come_from_the_result_files():
    cleaning = {"raw_videos": 3, "raw_comments": 40, "kept": 30}
    agreement = {"claude_vs_gemini": {"overall": {"cohen_kappa": 0.5}}}
    nums = mrf.pipeline_numbers(cleaning, agreement, DS)
    assert nums == {"videos": 3, "raw": 40, "clean": 30, "kappa": 0.5, "train": 20, "val": 5, "test": 5}


def test_class_rows_order_languages_then_splits():
    assert [r[0] for r in mrf.class_rows(DS)] == ["Hindi", "Bengali", "train", "val", "test"]


def test_robustness_bars_read_the_single_run():
    run = {"test_macro_f1": 0.7, "noisy_macro_f1": 0.6, "challenge_macro_f1": 0.8,
           "challenge_by_phenomenon": {k: {"macro_f1": 0.5} for k in ("negation", "sarcasm", "implicit",
                                                                       "topic_not_target")}}
    bars = mrf.robustness_bars({"plain": {"runs": [run]}})
    assert [b[1] for b in bars[:3]] == [0.7, 0.6, 0.8] and len(bars) == 7


def test_figures_are_written(tmp_path):
    assert mrf.fig_script_mix(DS, tmp_path).read_bytes()[:4] == b"\x89PNG"
    assert mrf.fig_class_dist(DS, tmp_path).exists()


def test_inline_markup_escapes_and_converts():
    out = br.inline("**Best** a < b, *note*, `x_y` and [site](https://e.org)")
    assert "<b>Best</b>" in out and "a &lt; b" in out and "<i>note</i>" in out
    assert '<font face="Courier"' in out and '<link href="https://e.org"' in out


def test_parse_blocks():
    md = """# Title

*Sub title*

<!-- a comment
over two lines -->

## 1. Part

Line one
line two.

- item one
  continued
- item two

| A | B |
|---|---:|
| x | 1 |

![a](figures/a.png "w=3") ![b](figures/b.png "w=2.5")

*Figure 1. Caption.*

---

Footer text.
"""
    blocks = br.parse(md)
    kinds = [b[0] for b in blocks]
    assert kinds == ["title", "subtitle", "h2", "p", "bullets", "table", "images", "caption", "rule", "footer"]
    assert blocks[1][1] == "Sub title" and blocks[3][1] == "Line one line two."
    assert blocks[4][1] == ["item one continued", "item two"]
    assert blocks[5][1] == [["A", "B"], ["x", "1"]] and blocks[5][2] == ["LEFT", "RIGHT"]
    assert blocks[6][1] == [("figures/a.png", 3.0), ("figures/b.png", 2.5)]


def test_build_pdf_counts_pages(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    blocks = br.parse("# T\n\n*S*\n\n## H\n\nSome text.\n")
    assert br.build_pdf(blocks, tmp_path / "t.pdf", 1.0, tmp_path) == 1


def test_col_widths_keep_short_columns_unwrapped_and_fill_the_width():
    rows = [["E0", "a much longer description " * 4, "another long text column " * 4],
            ["E1", "x", "y"]]
    widths = br._col_widths(rows, 500.0, 3, font_size=9.0)
    assert abs(sum(widths) - 500.0) < 1e-6
    assert widths[0] == (len("E0") + 2) * 0.5 * 9.0  # natural width, no wrapping
    assert widths[1] > widths[0] and widths[2] > widths[0]


def test_subtitle_lines_are_all_paragraphs_before_the_first_heading():
    kinds = [b[0] for b in br.parse("# T\n\nline one\n\nline two\n\n## H\n\ntext\n")]
    assert kinds == ["title", "subtitle", "subtitle", "h2", "p"]
