"""Tests for terralens.data external loaders."""

import pandas as pd
import pytest

from terralens.data import GOLD_CHECK_FILES, load_all_external


def _write(path, n=2):
    pd.DataFrame({
        "id": [f"x{i}" for i in range(n)], "text": ["t"] * n, "label": ["Favour"] * n,
        "source": ["s"] * n, "lang": ["en"] * n,
    }).to_csv(path, index=False)


def test_load_all_external_refuses_gold_check_set_by_default(tmp_path):
    assert "climatestance_gold" in GOLD_CHECK_FILES
    _write(tmp_path / "climatestance_gold.csv")
    with pytest.raises(ValueError, match="gold check"):
        load_all_external(["climatestance_gold"], external_dir=tmp_path)


def test_load_all_external_allows_gold_check_set_when_explicit(tmp_path):
    _write(tmp_path / "climatestance_gold.csv")
    _write(tmp_path / "gwsd.csv", n=3)
    df = load_all_external(["gwsd", "climatestance_gold"], external_dir=tmp_path, allow_gold_check=True)
    assert len(df) == 5
