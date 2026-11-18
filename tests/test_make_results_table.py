"""Tests for scripts/make_results_table.py."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mrt", ROOT / "scripts" / "make_results_table.py")
mrt = importlib.util.module_from_spec(_spec)
sys.modules["mrt"] = mrt
_spec.loader.exec_module(mrt)


def _m(f):
    return {"macro_f1": f, "against_f1": f / 2, "accuracy": f, "ci95": {"macro_f1": [f - 0.1, f + 0.1]},
            "by_lang": {"hi": {"macro_f1": f}}, "by_script": {}}


def _enc(f, seed=42):
    return {"kind": "encoder", "seed": seed, "config": {"model_name": "m"}, "val": _m(f), "test": _m(f - 0.05),
            "thresholds": {"test_tuned": _m(f)}}


def test_missing_planned_experiments_are_pending():
    rows = mrt.main_rows({"e2_mmbert": _enc(0.6)})
    by_id = {r[1]: r for r in rows}
    assert by_id["e1_qwen3_8b_0shot"][3] == "pending"
    assert by_id["e2_mmbert"][4] == "0.600" and "[0.450, 0.650]" in by_id["e2_mmbert"][6]


def test_best_by_val_ignores_seed_runs_and_seed_table():
    res = {"e2_a": _enc(0.5), "e2_b": _enc(0.6), "e2_b_s43": _enc(0.9, 43), "e2_b_s44": _enc(0.7, 44)}
    assert mrt.best_by_val(res, "e2_") == "e2_b"
    rows = mrt.seed_rows(res)
    assert len(rows) == 1 and rows[0][0] == "e2_b" and rows[0][2].startswith("0.733 ±")
