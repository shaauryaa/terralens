"""TF-IDF (word 1-2 + char 2-5 n-grams) + LogisticRegression / LinearSVC baselines,
evaluated with stratified k-fold cross-validation.

Usage:
    python -m terralens.baselines --config configs/baseline.yaml
    python -m terralens.baselines --config configs/baseline.yaml --dataset semeval_climate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from terralens.data import RAW_DIR, label_counts, load_external, load_official
from terralens.evaluate import evaluate, format_report, mean_std
from terralens.preprocess import preprocess_df

DEFAULT_CONFIG = {
    "seed": 42,
    "features": {
        "word": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000, "sublinear_tf": True},
        "char": {
            "analyzer": "char_wb",
            "ngram_range": [2, 5],
            "min_df": 2,
            "max_features": 100000,
            "sublinear_tf": True,
        },
    },
    "models": {
        "logreg": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
        "linear_svc": {"C": 0.5, "class_weight": "balanced"},
    },
    "cv": {"n_splits": 5, "shuffle": True},
}


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(cfg: dict) -> FeatureUnion:
    word = dict(cfg["word"])
    char = dict(cfg["char"])
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(word.pop("ngram_range")),
                    token_pattern=r"(?u)\b\w+\b",
                    lowercase=False,  # lower-casing is handled in preprocessing
                    **word,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer=char.pop("analyzer", "char_wb"),
                    ngram_range=tuple(char.pop("ngram_range")),
                    lowercase=False,
                    **char,
                ),
            ),
        ]
    )


def build_model(name: str, config: dict) -> Pipeline:
    params = dict(config["models"][name])
    seed = config.get("seed", 42)
    if name == "logreg":
        clf = LogisticRegression(random_state=seed, **params)
    elif name == "linear_svc":
        clf = LinearSVC(random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("tfidf", build_features(config["features"])), ("clf", clf)])


def cross_validate(df: pd.DataFrame, model_name: str, config: dict, text_col: str = "clean_text") -> dict:
    """Stratified k-fold CV. Returns per-fold scores plus metrics on pooled out-of-fold predictions."""
    cv_cfg = config["cv"]
    seed = config.get("seed", 42)
    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg.get("shuffle", True),
        random_state=seed if cv_cfg.get("shuffle", True) else None,
    )
    X = df[text_col].to_numpy()
    y = df["label"].to_numpy()
    oof = np.empty(len(df), dtype=object)
    fold_macro, fold_against = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = build_model(model_name, config)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        oof[va] = pred
        res = evaluate(y[va], pred)
        fold_macro.append(res["macro_f1"])
        fold_against.append(res["against_f1"])
        print(f"  [{model_name}] fold {fold}: macro F1 = {res['macro_f1']:.4f}, Against F1 = {res['against_f1']:.4f}")

    groups = df["script"] if "script" in df.columns else None
    return {
        "fold_macro_f1": fold_macro,
        "fold_against_f1": fold_against,
        "oof_pred": oof,
        "oof_eval": evaluate(y, oof, groups=groups),
    }


def load_dataset_for_cv(config: dict, dataset: str | None = None) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    name = dataset or data_cfg.get("dataset", "official")
    if name == "official":
        df = load_official(RAW_DIR / data_cfg.get("train_file", "train.csv"))
    else:
        df = load_external(name)
    df = df.dropna(subset=["label"])
    return preprocess_df(df, text_col=data_cfg.get("text_column", "text"), lowercase=data_cfg.get("lowercase", True))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dataset", default=None, help="override data.dataset (official | semeval_climate | gwsd)")
    parser.add_argument("--models", nargs="+", default=None, help="subset of models in the config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = load_dataset_for_cv(config, args.dataset)
    print(f"Loaded {len(df)} labelled rows. Class counts:\n{label_counts(df).to_string()}\n")

    summary = []
    for name in args.models or list(config["models"]):
        print(f"== {name} ({config['cv']['n_splits']}-fold stratified CV) ==")
        res = cross_validate(df, name, config)
        m_mean, m_std = mean_std(res["fold_macro_f1"])
        a_mean, a_std = mean_std(res["fold_against_f1"])
        print(f"  mean macro F1 = {m_mean:.4f} +/- {m_std:.4f}; mean Against F1 = {a_mean:.4f} +/- {a_std:.4f}")
        print("\nOut-of-fold report:\n" + format_report(res["oof_eval"]) + "\n")
        summary.append({"model": name, "macro_f1_mean": m_mean, "macro_f1_std": m_std,
                        "against_f1_mean": a_mean, "against_f1_std": a_std})

    print("Summary:\n" + pd.DataFrame(summary).set_index("model").round(4).to_string())


if __name__ == "__main__":
    main()
