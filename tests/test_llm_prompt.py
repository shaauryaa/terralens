"""Tests for the CPU-side parts of terralens.models.llm_prompt."""

from pathlib import Path

import pandas as pd
import pytest

from terralens.models.llm_prompt import build_messages, choose_shots, load_config, parse_label

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("answer, expected", [
    ("Favour", "Favour"), ("favor.", "Favour"), ("  Against", "Against"), ("None", "None"),
    ("Neutral", "None"), ("<think>Against?</think>\nFavour", "Favour"), ("Against, not Favour", "Against"),
    ("I cannot tell", None), ("", None),
])
def test_parse_label(answer, expected):
    assert parse_label(answer) == expected


def _train():
    rows = []
    for lab in ("Favour", "Against", "None"):
        for lang in ("hi", "bn"):
            for i in range(4):
                rows.append({"id": f"{lab}_{lang}_{i}", "text": f"word word word {lab} {i}", "label": lab, "lang": lang})
    rows.append({"id": "long", "text": "x " * 100, "label": "Against", "lang": "hi"})
    return pd.DataFrame(rows)


def test_choose_shots_follows_quota_and_is_deterministic():
    tr = _train()
    s = choose_shots(tr, {"Favour": 2, "Against": 2, "None": 1}, max_tokens=40, seed=42)
    assert s["label"].value_counts().to_dict() == {"Favour": 2, "Against": 2, "None": 1}
    assert "long" not in set(s["id"]) and s["id"].is_unique
    assert set(s.loc[s["label"] == "Favour", "lang"]) == {"hi", "bn"}
    assert s["id"].tolist() == choose_shots(tr, {"Favour": 2, "Against": 2, "None": 1}, seed=42)["id"].tolist()


def test_build_messages_alternates_user_and_assistant():
    shots = pd.DataFrame({"text": ["a b c", "d e f"], "label": ["Favour", "None"]})
    msgs = build_messages("new comment", shots)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "Favour" and msgs[-1]["content"] == "Comment: new comment"
    assert len(build_messages("x", None)) == 2


@pytest.mark.parametrize("name", ["e1_qwen3_8b_0shot", "e1_qwen3_8b_5shot", "e1_gemma3_12b_0shot",
                                  "e1_gemma3_12b_5shot"])
def test_e1_configs(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["exp_id"] == name and cfg["load_in_4bit"]
    assert cfg["shots"] == (5 if name.endswith("5shot") else 0)
    assert not any(a in cfg["model_name"].lower() for a in ("claude", "gemini", "gpt-oss"))
