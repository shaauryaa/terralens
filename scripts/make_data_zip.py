"""Bundle the data a GPU run needs into one zip (for Google Drive / Colab or a private Kaggle dataset).

Contents (paths inside the zip mirror the repo):
    data/processed/source_a.csv
    data/splits/{train,val,test}_ids.txt
    data/processed/{translated,romanised,synthetic}.csv     if they exist (training-only data)
    data/challenge/*.csv                                     if they exist
    data/processed/test_noisy.csv                            if it exists

The zip holds comment text, so it must stay private (never commit it or share it publicly).

Usage:
    python scripts/make_data_zip.py                       # -> outputs/terralens_data.zip
    python scripts/make_data_zip.py --out D:/drive/terralens_data.zip
    python scripts/make_data_zip.py --limit 1             # dry run: list files, write nothing
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["data/processed/source_a.csv", "data/splits/train_ids.txt", "data/splits/val_ids.txt",
            "data/splits/test_ids.txt"]
OPTIONAL_GLOBS = ["data/processed/translated.csv", "data/processed/romanised.csv", "data/processed/synthetic.csv",
                  "data/processed/test_noisy.csv", "data/challenge/*.csv"]


def collect(root: Path = ROOT) -> list[Path]:
    files = []
    for rel in REQUIRED:
        path = root / rel
        if not path.exists():
            raise SystemExit(f"missing {rel}; run make_source_a.py and make_splits.py first")
        files.append(path)
    for pattern in OPTIONAL_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    return files


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "terralens_data.zip")
    p.add_argument("--limit", type=int, default=None, help="dry run: list the files only")
    args = p.parse_args(argv)
    files = collect()
    for f in files:
        print(f"  {f.relative_to(ROOT).as_posix()}  ({f.stat().st_size / 1e6:.2f} MB)")
    if args.limit:
        print("Dry run: nothing written.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
