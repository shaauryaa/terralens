"""Tests for terralens.evaluate (bootstrap CIs and JSON metrics)."""

from terralens.evaluate import bootstrap_ci, macro_f1, metrics_json


def _data():
    y = ["Favour"] * 30 + ["Against"] * 10 + ["None"] * 60
    p = ["Favour"] * 25 + ["None"] * 5 + ["Against"] * 7 + ["None"] * 3 + ["None"] * 55 + ["Favour"] * 5
    return y, p


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    y, p = _data()
    ci = bootstrap_ci(y, p, n_resamples=300, seed=1)
    m = macro_f1(y, p)
    assert ci["macro_f1"][0] <= m <= ci["macro_f1"][1]
    assert ci["against_f1"][0] < ci["against_f1"][1]
    assert ci == bootstrap_ci(y, p, n_resamples=300, seed=1)


def test_bootstrap_ci_perfect_predictions():
    y = ["Favour", "Against", "None"] * 10
    ci = bootstrap_ci(y, y, n_resamples=50)
    assert ci["macro_f1"] == [1.0, 1.0]


def test_metrics_json_has_all_fields():
    y, p = _data()
    langs = ["hi", "bn"] * 50
    out = metrics_json(y, p, groups={"lang": langs}, n_boot=50)
    assert out["n"] == 100 and 0 <= out["macro_f1"] <= 1
    assert set(out["per_class"]) == {"Favour", "Against", "None"}
    assert sum(map(sum, out["confusion_matrix"]["matrix"])) == 100
    assert set(out["by_lang"]) == {"hi", "bn"} and out["ci95"]["n_resamples"] == 50
    assert out["accuracy"] == round((25 + 7 + 55) / 100, 4)
