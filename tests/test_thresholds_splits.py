"""Tests for terralens.thresholds and terralens.splits."""

import numpy as np
import pandas as pd

from terralens.splits import load_splits, parse_tags
from terralens.thresholds import predict_labels, tune_offsets


def test_tune_offsets_recovers_a_shifted_class():
    rng = np.random.default_rng(0)
    y = ["Favour"] * 40 + ["Against"] * 10 + ["None"] * 50
    idx = {"Favour": 0, "Against": 1, "None": 2}
    logits = rng.normal(0, 0.3, (100, 3))
    for i, lab in enumerate(y):
        logits[i, idx[lab]] += 1.0
    logits[:, 1] -= 1.5  # Against is under-predicted
    before = predict_labels(logits)
    off, f = tune_offsets(logits, y, grid=np.arange(-2, 2.01, 0.25))
    after = predict_labels(logits, off)
    assert off[0] == 0.0 and off[1] > 0
    assert after.count("Against") > before.count("Against")
    assert f >= 0.9


def test_tune_offsets_prefers_zero_when_already_perfect():
    logits = np.eye(3)[[0, 1, 2, 0, 1, 2]] * 5
    off, f = tune_offsets(logits, ["Favour", "Against", "None"] * 2, grid=[-1, 0, 1])
    assert off == [0.0, 0.0, 0.0] and f == 1.0


def test_load_splits_keeps_extra_data_out_of_val_and_test(tmp_path):
    a = pd.DataFrame({"id": list("abcdef"), "text": ["one two", "three four", "five six", "seven eight",
                                                     "nine ten", "eleven twelve"],
                      "label": ["Favour", "None", "Against", "None", "Favour", "None"], "lang": ["hi"] * 6,
                      "script": ["latin"] * 6, "source": ["youtube"] * 6, "video_id": list("vvwwxx"),
                      "tags": ["[]"] * 6})
    a.to_csv(tmp_path / "a.csv", index=False)
    sd = tmp_path / "splits"
    sd.mkdir()
    (sd / "train_ids.txt").write_text("a\nb\n")
    (sd / "val_ids.txt").write_text("c\nd\n")
    (sd / "test_ids.txt").write_text("e\nf\n")
    extra = pd.DataFrame({"id": ["x1", "x2", "c"], "text": ["new text here", "nine ten", "anything"],
                          "label": ["Against", "Favour", "None"], "lang": ["hi"] * 3, "source": ["synthetic"] * 3})
    extra.to_csv(tmp_path / "extra.csv", index=False)
    s = load_splits(tmp_path / "a.csv", sd, extra_train=[tmp_path / "extra.csv"])
    assert s["val"]["id"].tolist() == ["c", "d"] and s["test"]["id"].tolist() == ["e", "f"]
    assert s["train"]["id"].tolist() == ["a", "b", "x1"]  # x2 matches test text, c matches a val id
    assert (s["val"]["source"] == "youtube").all()
    assert s["train"].loc[s["train"]["id"] == "x1", "script"].item() == "latin"


def test_parse_tags():
    assert parse_tags('["sarcasm", "negation"]') == ["sarcasm", "negation"]
    assert parse_tags("") == [] and parse_tags("not json") == []
