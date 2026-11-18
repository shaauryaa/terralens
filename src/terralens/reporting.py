"""Shared helpers for experiment result files (results/<experiment_id>.json) and prediction files."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from terralens import LABELS
from terralens.evaluate import against_f1, macro_f1, metrics_json
from terralens.splits import parse_tags

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PRED_DIR = ROOT / "results" / "predictions"
TAGS = ("sarcasm", "negation", "implicit", "code_mixed", "romanised", "question")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_metrics(df: pd.DataFrame, pred, n_boot: int = 1000, seed: int = 42) -> dict:
    """metrics_json with per-language and per-script breakdowns, plus Macro F1 per annotator tag."""
    pred = list(pred)
    out = metrics_json(df["label"].tolist(), pred, groups={"lang": df["lang"].tolist(),
                                                           "script": df["script"].tolist()},
                       n_boot=n_boot, seed=seed)
    tags = df["tags"].map(parse_tags).tolist() if "tags" in df else [[] for _ in pred]
    by_tag = {}
    y = df["label"].tolist()
    for tag in TAGS:
        idx = [i for i, t in enumerate(tags) if tag in t]
        if idx:
            yt, yp = [y[i] for i in idx], [pred[i] for i in idx]
            by_tag[tag] = {"n": len(idx), "macro_f1": round(macro_f1(yt, yp), 4),
                           "against_f1": round(against_f1(yt, yp), 4),
                           "accuracy": round(float(np.mean(np.array(yt) == np.array(yp))), 4)}
    out["by_tag"] = by_tag
    return out


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def write_result(exp_id: str, payload: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{exp_id}.json"
    body = {"experiment_id": exp_id, "timestamp": now(), **payload}
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")
    return path


def write_predictions(exp_id: str, split: str, df: pd.DataFrame, pred, scores=None, pred_dir: Path = PRED_DIR,
                      extra: dict | None = None) -> Path:
    """id, gold, pred (+ one score column per label) - no comment text, so the file can be shared."""
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": df["id"].values, "gold": df["label"].values, "pred": list(pred),
                        "lang": df["lang"].values, "script": df["script"].values})
    if scores is not None:
        scores = np.asarray(scores)
        for j, lab in enumerate(LABELS):
            out[f"score_{lab}"] = np.round(scores[:, j], 5)
    for k, v in (extra or {}).items():
        out[k] = list(v)
    path = pred_dir / f"{exp_id}_{split}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
