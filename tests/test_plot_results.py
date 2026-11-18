"""Tests for scripts/plot_results.py."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("plot_results", ROOT / "scripts" / "plot_results.py")
pr = importlib.util.module_from_spec(_spec)
sys.modules["plot_results"] = pr
_spec.loader.exec_module(pr)


def _write(d: Path, name: str, kind: str, val: float, test: float) -> None:
    r = {"kind": kind, "val": {"macro_f1": val},
         "test": {"macro_f1": test, "ci95": {"macro_f1": [test - 0.05, test + 0.05]}}}
    (d / f"{name}.json").write_text(json.dumps(r), encoding="utf-8")


def test_collect_keeps_systems_only_in_fixed_order(tmp_path):
    _write(tmp_path, "e4_qlora", "qlora", 0.7, 0.68)
    _write(tmp_path, "e2_mmbert", "encoder", 0.66, 0.63)
    _write(tmp_path, "e0_majority", "majority", 0.3, 0.27)
    _write(tmp_path, "e2_mmbert_s43", "encoder", 0.9, 0.9)      # seed rerun: left out
    _write(tmp_path, "e4_qlora_robust", "qlora", 0.9, 0.9)      # robustness rerun: left out
    _write(tmp_path, "robustness_e4_qlora", "robustness", 0, 0)  # not a system
    (tmp_path / "splits.json").write_text("{}", encoding="utf-8")
    rows = pr.collect_systems(tmp_path)
    assert [d["exp_id"] for d in rows] == ["e0_majority", "e2_mmbert", "e4_qlora"]
    assert rows[1]["ci_low"] == 0.58 and rows[1]["group"] == "E2"


def test_best_is_chosen_on_val_not_test(tmp_path):
    _write(tmp_path, "e1_qwen3_8b_5shot", "llm_prompt", 0.52, 0.70)  # best on test
    _write(tmp_path, "e2_mmbert", "encoder", 0.66, 0.63)             # best on val
    _write(tmp_path, "e0_majority", "majority", 0.99, 0.27)          # floor never counts
    assert pr.best_by_val(pr.collect_systems(tmp_path)) == "e2_mmbert"


def test_plot_writes_png(tmp_path):
    _write(tmp_path, "e0_majority", "majority", 0.3, 0.27)
    _write(tmp_path, "e2_mmbert", "encoder", 0.66, 0.63)
    out = pr.plot(pr.collect_systems(tmp_path), tmp_path / "fig.png")
    assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"
