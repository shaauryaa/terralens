"""Tests for the CPU-side parts of terralens.models.encoder and scripts/run_majority.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from terralens.models.encoder import class_weights, load_config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_majority", ROOT / "scripts" / "run_majority.py")
rm = importlib.util.module_from_spec(_spec)
sys.modules["run_majority"] = rm
_spec.loader.exec_module(rm)


def test_class_weights_are_inverse_frequency():
    w = class_weights(["Favour"] * 2 + ["Against"] * 1 + ["None"] * 7)
    assert np.allclose(w, [10 / 6, 10 / 3, 10 / 21])
    assert w[1] > w[0] > w[2]


@pytest.mark.parametrize("name", ["e2_xlmr", "e2_muril", "e2_indicbertv2", "e2_mmbert"])
def test_e2_configs_follow_the_protocol(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["exp_id"] == name
    assert cfg["max_length"] == 128 and cfg["lr"] == 2e-5 and cfg["batch_size"] == 32 and cfg["epochs"] == 4
    assert cfg["warmup_ratio"] == 0.1 and cfg["fp16"] and cfg["class_weighted"] and cfg["seed"] == 42
    assert cfg["extra_train"] == []


def test_load_config_overrides_and_requires_ids(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("exp_id: x\nmodel_name: m\nepochs: 4\n")
    cfg = load_config(p, {"seed": 43, "epochs": None})
    assert cfg["seed"] == 43 and cfg["epochs"] == 4
    (tmp_path / "bad.yaml").write_text("model_name: m\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.yaml")


def test_majority_label_ties_break_alphabetically():
    assert rm.majority_label(["None", "Favour", "None"]) == "None"
    assert rm.majority_label(["None", "Favour"]) == "Favour"
