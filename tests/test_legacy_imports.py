"""Smoke test: the legacy TF-IDF code still imports and builds after the move to legacy/."""

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from legacy.baselines import build_model, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_baseline_builds_from_config():
    config = load_config(ROOT / "legacy" / "configs" / "baseline.yaml")
    model = build_model("logreg", config)
    assert [name for name, _ in model.steps] == ["tfidf", "clf"]
