"""Tests for scripts/run_final.py (selection by validation only)."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_final", ROOT / "scripts" / "run_final.py")
rf = importlib.util.module_from_spec(_spec)
sys.modules["run_final"] = rf
_spec.loader.exec_module(rf)


def test_choose_uses_val_not_test(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "RESULTS", tmp_path)
    cfgs = []
    for name, val, test in (("a", 0.6, 0.9), ("b", 0.7, 0.1)):
        c = tmp_path / f"{name}.yaml"
        c.write_text(f"exp_id: {name}\n")
        cfgs.append(c)
        (tmp_path / f"{name}.json").write_text(json.dumps({"kind": "encoder", "val": {"macro_f1": val, "against_f1": 0.1},
                                                           "test": {"macro_f1": test}}))
    best, scores = rf.choose(cfgs)
    assert best.name == "b.yaml" and len(scores) == 2
