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
