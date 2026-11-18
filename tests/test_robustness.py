"""Tests for terralens.robustness."""

from terralens.robustness import evaluate_robustness


def _res(seed, test_f1, noisy_f1, ch_f1):
    m = lambda f: {"macro_f1": f, "against_f1": f / 2, "n": 10, "by_tag": {"negation": {"n": 5, "macro_f1": f,  # noqa: E731
                                                                                       "accuracy": f}}, "by_lang": {}}
    return {"experiment_id": f"x_s{seed}", "seed": seed, "test": m(test_f1),
            "thresholds": {"test_tuned": m(test_f1)},
            "extra_eval": {"test_noisy": {"plain": m(noisy_f1), "tuned": m(noisy_f1)},
                           "challenge": {"plain": m(ch_f1), "tuned": m(ch_f1)}}}


def test_evaluate_robustness_drop_and_mean_std():
    out = evaluate_robustness([_res(42, 0.6, 0.5, 0.4), _res(43, 0.7, 0.5, 0.6)])
    assert [r["drop_macro_f1"] for r in out["runs"]] == [0.1, 0.2]
    ms = out["mean_std"]
    assert ms["drop_macro_f1"]["mean"] == 0.15 and ms["challenge_macro_f1"]["n_runs"] == 2
    assert ms["challenge_by_phenomenon"]["negation"]["mean"] == 0.5
