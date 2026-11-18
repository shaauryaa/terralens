"""Tests for scripts/error_analysis.py (sampling only; no API calls)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("error_analysis", ROOT / "scripts" / "error_analysis.py")
ea = importlib.util.module_from_spec(_spec)
sys.modules["error_analysis"] = ea
_spec.loader.exec_module(ea)


def test_misclassified_sample_only_errors_and_covers_pairs():
    gold = ["Favour"] * 30 + ["None"] * 60 + ["Against"] * 10
    pred = ["None"] * 10 + ["Favour"] * 20 + ["Favour"] * 30 + ["None"] * 30 + ["Favour"] * 5 + ["Against"] * 5
    val = pd.DataFrame({"id": [str(i) for i in range(100)], "label": gold})
    p = pd.DataFrame({"id": val["id"], "pred": pred})
    s = ea.misclassified_sample(val, p, n=20)
    assert len(s) == 20 and (s["label"] != s["pred"]).all()
    assert set(s["pair"]) == {"Favour->None", "None->Favour", "Against->Favour"}
    assert s["id"].tolist() == ea.misclassified_sample(val, p, n=20)["id"].tolist()
