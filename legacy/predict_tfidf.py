"""LEGACY. Train a TF-IDF baseline on the full training file and write a predictions file (``id,label``).

Usage (from the repo root):
    python -m legacy.predict_tfidf --train data/raw/train.csv --test data/raw/test.csv \
        --model logreg --out outputs/predictions_tfidf.csv
"""

from __future__ import annotations

import argparse

from legacy.baselines import build_model, load_config
from terralens.data import load_official
from terralens.predict import write_predictions
from terralens.preprocess import preprocess_df


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="legacy/configs/baseline.yaml")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="logreg", choices=["logreg", "linear_svc"])
    parser.add_argument("--out", default="outputs/predictions_tfidf.csv")
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
