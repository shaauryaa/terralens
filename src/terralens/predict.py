"""Train a baseline on the full training file and write a predictions file (``id,label``).

Usage:
    python -m terralens.predict --train data/raw/train.csv --test data/raw/test.csv \
        --model logreg --out outputs/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from terralens import LABELS
from terralens.baselines import build_model, load_config
from terralens.data import load_official
from terralens.preprocess import preprocess_df


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="logreg", choices=["logreg", "linear_svc"])
    parser.add_argument("--out", default="outputs/predictions.csv")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    lowercase = config.get("data", {}).get("lowercase", True)
    train = preprocess_df(load_official(args.train), lowercase=lowercase)
    test = preprocess_df(load_official(args.test, require_labels=False), lowercase=lowercase)

    model = build_model(args.model, config)
    model.fit(train["clean_text"], train["label"])
    path = write_predictions(test["id"], model.predict(test["clean_text"]), args.out)
    print(f"Wrote {len(test)} predictions to {path}")


if __name__ == "__main__":
    main()
