"""Tests for the label mapping, exclusion and dedup logic in scripts/download_external.py. No network calls."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("download_external", ROOT / "scripts" / "download_external.py")
dx = importlib.util.module_from_spec(_spec)
sys.modules["download_external"] = dx  # dataclasses look the module up while the class is created
_spec.loader.exec_module(dx)

TERRALENS_LABELS = {"Favour", "Against", "None"}


def test_mapping_tables_cover_expected_codes_and_use_valid_labels():
    assert set(dx.CLIMATESTANCE_MAP) == {1, 0, -1}
    assert set(dx.KAGGLE_MAP) == {2, 1, 0, -1}
    for mapping in (dx.CLIMATESTANCE_MAP, dx.KAGGLE_MAP, dx.SEMEVAL_MAP):
        for rule in mapping.values():
            assert rule.label in TERRALENS_LABELS
            assert rule.confidence in {"direct", "approximate", "uncertain"}
            assert rule.reason
    assert dx.CLIMATESTANCE_MAP[1].label == "Favour"
    assert dx.CLIMATESTANCE_MAP[-1].label == "Against"
    assert dx.CLIMATESTANCE_MAP[0].label == "None"
    assert [dx.KAGGLE_MAP[c].label for c in (1, -1, 0, 2)] == ["Favour", "Against", "None", "None"]


def test_dedup_key_ignores_case_urls_mentions_and_retweet_prefix():
    a = dx.dedup_key("RT @someone: Climate change is REAL https://t.co/abc")
    b = dx.dedup_key("climate change is real")
    assert a == b == "climate change is real"
    assert dx.dedup_key("   ") == ""


def test_apply_mapping_keeps_original_and_excludes_unknown_codes():
    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "code": [1, -1, 7, None]})
    kept, excluded = dx.apply_mapping(df, "code", dx.CLIMATESTANCE_MAP)
    assert kept["label"].tolist() == ["Favour", "Against"]
    assert kept["original_label_name"].tolist() == ["favor", "against"]
    assert kept["mapping_confidence"].tolist() == ["direct", "approximate"]
    assert len(excluded) == 2 and set(excluded["reason"]) == {"unknown_label"}


def test_dedup_within_keeps_first_and_excludes_conflicts():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "text": ["Same text", "same  TEXT", "Clash", "clash", "unique"],
        "label": ["Favour", "Favour", "Favour", "Against", "None"],
    })
    kept, excluded = dx.dedup_within(df)
    assert kept["id"].tolist() == ["1", "5"]
    reasons = dict(zip(excluded["id"], excluded["reason"]))
    assert reasons == {"2": "duplicate_text", "3": "duplicate_conflicting_labels", "4": "duplicate_conflicting_labels"}


def _reddit(opinions):
    return pd.DataFrame({
        "Unnamed: 0": range(len(opinions)),
        "permalink": [f"/r/climate/comments/x{i}/t/c{i}/" for i in range(len(opinions))],
        "body": ["warming is scary", "it is a hoax", "nice weather", "", "it is a hoax"][: len(opinions)],
        "topic-label": [1] * len(opinions),
        "opinion-label": opinions,
    })


def test_climatestance_gold_reports_disagreements_and_empty_text():
    a1 = _reddit([1, -1, 0, 1, -1])
    a2 = _reddit([1, -1, 1, 1, -1])
    built = dx.build_climatestance_gold(a1, a2)
    assert built.kept["label"].tolist() == ["Favour", "Against"]
    assert set(built.kept["source_dataset"]) == {"climatestance"}
    assert list(built.kept.columns) == dx.OUTPUT_COLUMNS
    assert set(built.kept["split"]) == {"gold_check"}  # never train / val / test
    assert built.kept["original_label"].tolist() == ["1", "-1"]
    assert built.kept["mapping_confidence"].tolist() == ["direct", "approximate"]
    reasons = sorted(built.excluded["reason"])
    assert reasons == ["annotator_disagreement", "duplicate_text", "empty_text"]
    disagreement = built.excluded[built.excluded["reason"] == "annotator_disagreement"].iloc[0]
    assert disagreement["original_label"] == "a1=0|a2=1"
    # nothing is lost: kept + excluded = input rows
    assert len(built.kept) + len(built.excluded) == len(a1)


def test_climatestance_gold_rejects_misaligned_annotator_files():
    a1 = _reddit([1, 0])
    a2 = _reddit([1, 0]).iloc[::-1]
    with pytest.raises(ValueError):
        dx.build_climatestance_gold(a1, a2)


def test_climatestance_tweet_ids_map_float_codes_and_flag_missing_text():
    frames = {
        f"tweets_{s}": pd.DataFrame({"tweet_id": [f"{s}1", f"{s}2"], "stance-label": [1.0, -1.0],
                                     "topic-label": [1, 2]})
        for s in ("train", "val", "test")
    }
    frames["tweets_test"].loc[1, "tweet_id"] = "train1"  # duplicate ID across splits
    built = dx.build_climatestance_tweet_ids(frames)
    assert built.counts_before == {"-1": 3, "1": 3}
    assert len(built.kept) == 5 and set(built.excluded["reason"]) == {"duplicate_tweet_id"}
    assert built.kept["text"].eq("").all()
    assert dx.stats_for(built)["has_text"] is False


def test_build_kaggle_maps_all_four_classes_and_validates_columns():
    raw = pd.DataFrame({
        "sentiment": [1, -1, 0, 2, 5],
        "message": ["we must act", "climate scam", "hmm ok", "News: ice melts https://t.co/x", "odd"],
        "tweetid": ["11", "12", "13", "14", "15"],
    })
    built = dx.build_kaggle(raw)
    assert built.kept["label"].tolist() == ["Favour", "Against", "None", "None"]
    assert built.kept["original_label_name"].tolist() == ["Pro", "Anti", "Neutral", "News"]
    assert built.kept["original_id"].tolist() == ["11", "12", "13", "14"]
    assert built.counts_before == {"-1": 1, "0": 1, "1": 1, "2": 1, "5": 1}
    assert built.excluded["reason"].tolist() == ["unknown_label"]
    with pytest.raises(ValueError, match="missing columns"):
        dx.build_kaggle(raw.rename(columns={"message": "text"}))


def test_count_mojibake_flags_cp1252_decoded_utf8_only():
    texts = pd.Series(["clean text…", "this is brokenâ€¦", "twiceÃ¢â‚¬Â¦", "café", "दुनिया"])
    assert dx.count_mojibake(texts) == 2


def test_kaggle_without_file_or_api_prints_manual_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dx, "kaggle_credentials_present", lambda: False)
    assert dx.download_kaggle(tmp_path, limit=None, dry_run=False, explicit=None) == []
    assert "put twitter_sentiment_data.csv in data/external/" in capsys.readouterr().out


def test_kaggle_reads_manual_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dx, "kaggle_credentials_present", lambda: False)
    pd.DataFrame({"sentiment": [1, -1], "message": ["a b c", "d e f"], "tweetid": ["1", "2"]}).to_csv(
        tmp_path / dx.KAGGLE_RAW_NAME, index=False)
    (built,) = dx.download_kaggle(tmp_path, limit=None, dry_run=True, explicit=None)
    assert built.kept["label"].tolist() == ["Favour", "Against"]


def test_climatestance_raw_reads_cached_files_without_network(tmp_path, monkeypatch):
    def no_network(url, timeout=60):
        raise AssertionError(f"network call to {url}")

    monkeypatch.setattr(dx, "http_get", no_network)
    for key, repo_path in dx.CLIMATESTANCE_FILES.items():
        header = "tweet_id\tstance-label\ttopic-label\n1\t1.0\t1\n" if key.startswith("tweets") else \
            "Unnamed: 0\tpermalink\tbody\ttopic-label\topinion-label\n0\t/r/climate/c/x/\thot\t1\t1\n"
        (tmp_path / dx.Path(repo_path).name).write_text(header, encoding="utf-8")
    frames = dx.load_climatestance_raw(tmp_path, cache=True)
    assert set(frames) == set(dx.CLIMATESTANCE_FILES)
    assert frames["tweets_train"]["tweet_id"].tolist() == ["1"]


def test_write_csv_does_not_overwrite_without_flag(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("keep me", encoding="utf-8")
    assert dx.write_csv(pd.DataFrame({"a": [1]}), path, overwrite=False) is False
    assert path.read_text(encoding="utf-8") == "keep me"
    assert dx.write_csv(pd.DataFrame({"a": [1]}), path, overwrite=True) is True


def test_update_results_pending_entry_never_replaces_built_stats(tmp_path):
    path = tmp_path / "r.json"
    args = dx.parse_args(["--datasets", "kaggle"])
    pending = {"k": {"status": "not run yet"}}
    dx.update_results(path, {}, {}, args, pending)
    assert dx.json.loads(path.read_text(encoding="utf-8"))["datasets"]["k"]["status"] == "not run yet"
    dx.update_results(path, {"k": {"status": "built", "rows_kept": 3}}, {}, args)
    dx.update_results(path, {}, {}, args, pending)
    assert dx.json.loads(path.read_text(encoding="utf-8"))["datasets"]["k"]["rows_kept"] == 3


def test_cross_overlap_counts_shared_texts():
    frames = {"a": pd.DataFrame({"text": ["Hot planet", "x y"]}), "b": pd.DataFrame({"text": ["hot  PLANET"]})}
    assert dx.cross_overlap(frames) == {"a|b": 1}
