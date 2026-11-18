"""Evaluation: macro F1 (primary), Against-class F1, per-class P/R/F1,
confusion matrix and per-language / per-script breakdown."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from terralens import LABELS


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def against_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=["Against"], average="macro", zero_division=0))


def per_class_report(y_true, y_pred, labels: Sequence[str] = LABELS) -> pd.DataFrame:
    """Precision / recall / F1 / support for each class."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=list(labels)
    )


def confusion_df(y_true, y_pred, labels: Sequence[str] = LABELS) -> pd.DataFrame:
    """Confusion matrix with gold labels as rows and predictions as columns."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        cm,
        index=pd.Index(labels, name="gold"),
        columns=pd.Index(labels, name="pred"),
    )


def group_breakdown(y_true, y_pred, groups, labels: Sequence[str] = LABELS) -> pd.DataFrame:
    """Macro F1 and Against F1 per group (e.g. language or script tag)."""
    frame = pd.DataFrame({"y_true": list(y_true), "y_pred": list(y_pred), "group": list(groups)})
    rows = []
    for name, g in frame.groupby("group", sort=True):
        rows.append(
            {
                "group": name,
                "n": len(g),
                "macro_f1": macro_f1(g.y_true, g.y_pred, labels),
                "against_f1": against_f1(g.y_true, g.y_pred),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def evaluate(y_true, y_pred, groups=None, labels: Sequence[str] = LABELS) -> dict:
    """All metrics in one dict. ``groups`` (optional) enables the per-language breakdown."""
    result = {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "against_f1": against_f1(y_true, y_pred),
        "per_class": per_class_report(y_true, y_pred, labels),
        "confusion_matrix": confusion_df(y_true, y_pred, labels),
    }
    if groups is not None:
        result["by_group"] = group_breakdown(y_true, y_pred, groups, labels)
    return result


def format_report(result: dict) -> str:
    """Human-readable text version of ``evaluate`` output."""
    parts = [
        f"Macro F1:   {result['macro_f1']:.4f}",
        f"Against F1: {result['against_f1']:.4f}",
        "",
        "Per-class:",
        result["per_class"].round(4).to_string(),
        "",
        "Confusion matrix (rows = gold, cols = pred):",
        result["confusion_matrix"].to_string(),
    ]
    if "by_group" in result:
        parts += ["", "By language / script:", result["by_group"].round(4).to_string()]
    return "\n".join(parts)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


# ============================================================================== bootstrap and JSON output


def bootstrap_ci(y_true, y_pred, n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05,
                 labels: Sequence[str] = LABELS) -> dict:
    """Percentile bootstrap (resampling comments with replacement) 95% CIs for Macro F1 and Against F1.

    Returns {"macro_f1": [lo, hi], "against_f1": [lo, hi], "n_resamples": n, "seed": seed}.
    """
    yt = np.asarray(list(y_true), dtype=object)
    yp = np.asarray(list(y_pred), dtype=object)
    n = len(yt)
    if n == 0:
        return {"macro_f1": [None, None], "against_f1": [None, None], "n_resamples": 0, "seed": seed}
    lab = list(labels)
    idx_map = {l: i for i, l in enumerate(lab)}
    t = np.array([idx_map.get(v, -1) for v in yt])
    p = np.array([idx_map.get(v, -1) for v in yp])
    rng = np.random.default_rng(seed)
    k = len(lab)
    a = idx_map.get("Against")
    macro, against = np.empty(n_resamples), np.empty(n_resamples)
    for b in range(n_resamples):
        s = rng.integers(0, n, n)
        tt, pp = t[s], p[s]
        f1s = np.empty(k)
        for c in range(k):
            tp = np.sum((tt == c) & (pp == c))
            fp = np.sum((tt != c) & (pp == c))
            fn = np.sum((tt == c) & (pp != c))
            denom = 2 * tp + fp + fn
            f1s[c] = 2 * tp / denom if denom else 0.0
        macro[b] = f1s.mean()
        against[b] = f1s[a] if a is not None else np.nan
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"macro_f1": [round(float(x), 4) for x in np.percentile(macro, q)],
            "against_f1": [round(float(x), 4) for x in np.percentile(against, q)],
            "n_resamples": n_resamples, "seed": seed}


def _round(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def metrics_json(y_true, y_pred, groups: dict | None = None, n_boot: int = 1000, seed: int = 42,
                 labels: Sequence[str] = LABELS) -> dict:
    """JSON-ready metrics: Macro F1, Against F1, accuracy, per-class P/R/F1, confusion matrix, bootstrap CIs and
    a breakdown for each grouping in ``groups`` (e.g. {"lang": [...], "script": [...]})."""
    y_true, y_pred = list(y_true), list(y_pred)
    res = evaluate(y_true, y_pred, labels=labels)
    pc = res["per_class"]
    out = {
        "n": len(y_true),
        "macro_f1": _round(res["macro_f1"]),
        "against_f1": _round(res["against_f1"]),
        "accuracy": _round(float(np.mean(np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)))
                           if y_true else None),
        "per_class": {lab: {"precision": _round(pc.loc[lab, "precision"]), "recall": _round(pc.loc[lab, "recall"]),
                            "f1": _round(pc.loc[lab, "f1"]), "support": int(pc.loc[lab, "support"])}
                      for lab in labels},
        "confusion_matrix": {"rows_gold_cols_pred": list(labels),
                             "matrix": res["confusion_matrix"].values.tolist()},
        "ci95": bootstrap_ci(y_true, y_pred, n_boot, seed, labels=labels) if n_boot else None,
    }
    for name, g in (groups or {}).items():
        bd = group_breakdown(y_true, y_pred, list(g), labels)
        out[f"by_{name}"] = {str(k): {"n": int(r["n"]), "macro_f1": _round(r["macro_f1"]),
                                      "against_f1": _round(r["against_f1"])} for k, r in bd.iterrows()}
    return out
