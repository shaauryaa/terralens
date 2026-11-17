import pandas as pd
import pytest

from terralens.predict import write_predictions


def test_write_predictions_roundtrip(tmp_path):
    out = write_predictions(["a", "b", "c"], ["Favour", "Against", "None"], tmp_path / "sub" / "pred.csv")
    df = pd.read_csv(out, keep_default_na=False)
    assert list(df.columns) == ["id", "label"]
    assert list(df["label"]) == ["Favour", "Against", "None"]


def test_write_predictions_rejects_bad_labels(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        write_predictions(["a"], ["favor"], tmp_path / "p.csv")


def test_write_predictions_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        write_predictions(["a", "a"], ["Favour", "None"], tmp_path / "p.csv")
