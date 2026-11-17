"""Write prediction files (``id,label``).

The command-line entry point (``python -m terralens.predict``) will load a fine-tuned model once one
exists (see ``terralens.models``). The old TF-IDF predictor lives in ``legacy/predict_tfidf.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from terralens import LABELS


def write_predictions(ids, labels, out_path: str | Path) -> Path:
    """Write a two-column CSV ``id,label``; validates labels against the label set."""
    preds = pd.DataFrame({"id": list(ids), "label": list(labels)})
    bad = sorted(set(preds["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Predictions contain labels outside {LABELS}: {bad}")
    if preds["id"].duplicated().any():
        raise ValueError("Duplicate ids in predictions")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
