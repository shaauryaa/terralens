"""Loading the official TerraLens CSVs and the auxiliary external datasets.

Every loader returns a DataFrame with at least ``id``, ``text`` and ``label``,
where ``label`` is one of ``Favour``, ``Against``, ``None`` (or missing for an
unlabelled test file).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from terralens import LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

COMMON_COLUMNS = ["id", "text", "label", "source", "lang"]

# Lower-cased spellings seen across datasets -> canonical label.
LABEL_ALIASES = {
    "favour": "Favour",
    "favor": "Favour",
    "for": "Favour",
    "pro": "Favour",
    "support": "Favour",
    "agree": "Favour",
    "against": "Against",
    "anti": "Against",
    "oppose": "Against",
    "disagree": "Against",
    "none": "None",
    "neutral": "None",
    "neither": "None",
}

# cardiffnlp/tweet_eval "stance_climate" integer labels.
TWEET_EVAL_STANCE = {0: "None", 1: "Against", 2: "Favour"}


def normalize_label(value) -> str | None:
    """Map a raw label (string or alias) to Favour / Against / None.

    Returns ``None`` (Python None) for missing values so unlabelled test rows survive.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    if key == "":
        return None
    if key not in LABEL_ALIASES:
        raise ValueError(f"Unknown stance label: {value!r}")
    return LABEL_ALIASES[key]


def load_official(path: str | Path, require_labels: bool = True) -> pd.DataFrame:
    """Load an official TerraLens CSV with columns ``id, text, label``.

    Set ``require_labels=False`` for a test file without gold labels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The official dataset has not been released yet; "
            "place it in data/raw/ when it is."
        )
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False, na_values=[""])
    required = ["id", "text"] + (["label"] if require_labels else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].map(normalize_label)
        if require_labels and df["label"].isna().any():
            raise ValueError(f"{path} has {df['label'].isna().sum()} rows without a label")
    df["source"] = "official"
    return df


def load_external(name: str, external_dir: str | Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Load one of the CSVs written by ``scripts/download_external.py``.

    ``name`` is the file stem, e.g. ``"semeval_climate"`` or ``"gwsd"``.
    """
    path = Path(external_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_external.py")
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False, na_values=[""])
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].map(normalize_label)
    return df


def load_all_external(names=("semeval_climate", "gwsd"), external_dir=EXTERNAL_DIR) -> pd.DataFrame:
    """Concatenate several external datasets into the common schema."""
    frames = [load_external(n, external_dir)[COMMON_COLUMNS] for n in names]
    return pd.concat(frames, ignore_index=True)


def label_counts(df: pd.DataFrame) -> pd.Series:
    """Class counts in the canonical label order."""
    return df["label"].value_counts().reindex(LABELS, fill_value=0)
