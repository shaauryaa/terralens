"""Per-class decision offsets added to logits (or log-probabilities), tuned on validation only.

prediction = argmax(logits + offsets). The Favour offset is fixed at 0; the Against and None offsets are
searched on a grid to maximise validation Macro F1. Ties go to the smallest total |offset|.
"""

from __future__ import annotations

import numpy as np

from terralens import LABELS
from terralens.evaluate import macro_f1


def apply_offsets(logits: np.ndarray, offsets) -> np.ndarray:
    return np.asarray(logits, dtype=float) + np.asarray(offsets, dtype=float)[None, :]


def predict_labels(logits: np.ndarray, offsets=None, labels=LABELS) -> list[str]:
    z = np.asarray(logits, dtype=float) if offsets is None else apply_offsets(logits, offsets)
    return [labels[i] for i in z.argmax(axis=1)]


def tune_offsets(logits: np.ndarray, y_true, grid=None, labels=LABELS) -> tuple[list[float], float]:
    """(offsets, val Macro F1 with them). Offsets are [0, a, n] in LABELS order (Favour, Against, None)."""
    grid = np.round(np.arange(-3.0, 3.0001, 0.1), 2) if grid is None else np.asarray(grid, dtype=float)
    logits = np.asarray(logits, dtype=float)
    y = list(y_true)
    best = (-1.0, float("inf"), [0.0, 0.0, 0.0])
    for a in grid:
        for n in grid:
            off = [0.0, float(a), float(n)]
            f = macro_f1(y, predict_labels(logits, off, labels), labels)
            size = abs(a) + abs(n)
            if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and size < best[1]):
                best = (f, size, off)
    return best[2], round(float(best[0]), 4)
