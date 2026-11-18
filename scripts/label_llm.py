"""Stance labelling with API annotators (T4).

Design: two full annotators (Claude, in a separate Claude Code session, and Gemini) label every item;
gpt-oss-120b on Groq is an adjudicator for Claude/Gemini disagreements and also labels the English gold set
and, if budget remains, a 300-comment sample.

Subcommands
    prepare     write the shared item files (both full annotators label exactly these rows):
                  data/interim/labelling_items.csv   comments_clean.csv (3,798) + other_language comments with
                                                     3+ tokens from comments_removed.csv (lang_uncertain=1)
                  data/interim/gold_items.csv        303 English gold rows + the Hindi / Bengali translations
                and, if missing, the translation template data/interim/gold_translations.csv
                (stratified 100-row subset of the gold set, seed 42; never overwritten once it exists)
    run         label a set with an API annotator:   --annotator gemini --set gold|main
    adjudicate  gpt-oss on (1) Claude/Gemini disagreements, (2) the English gold rows, (3) a 300-comment
                random sample (seed 42), in that order, until --max-requests or the daily limit
    validate    check an annotator's label files (schema, coverage, duplicates), e.g. --annotator claude

Output (one JSON object per line, appended as soon as a batch is validated, so runs resume):
    data/interim/labels_<annotator>.jsonl        main items
    data/interim/labels_<annotator>_gold.jsonl   gold items
    data/interim/labels_<annotator>_failures.jsonl, labels_<annotator>_requests.jsonl (logs)
Each label line: {"id", "label", "confidence", "tags", ["lang"], "annotator", "model", "set", "ts"}.
Items that still fail validation after 3 attempts are logged as failures and stay unlabelled; a re-run
retries them. Nothing is dropped silently: `validate` reports every missing id.

Usage:
    python scripts/label_llm.py prepare
    python scripts/label_llm.py run --annotator gemini --set gold
    python scripts/label_llm.py run --annotator gemini --set main --max-requests 400
    python scripts/label_llm.py run --annotator gemini --set main --limit 10   # dry run (*.dryrun.jsonl)
    python scripts/label_llm.py adjudicate --max-requests 100
    python scripts/label_llm.py validate --annotator gemini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402
from terralens.preprocess import clean_text, tag_script  # noqa: E402

SEED = 42
INTERIM = ROOT / "data" / "interim"
CLEAN_CSV = INTERIM / "comments_clean.csv"
REMOVED_CSV = INTERIM / "comments_removed.csv"
GOLD_CSV = ROOT / "data" / "external" / "climatestance_gold.csv"
ITEMS_CSV = INTERIM / "labelling_items.csv"
GOLD_ITEMS_CSV = INTERIM / "gold_items.csv"
TRANSLATIONS_CSV = INTERIM / "gold_translations.csv"
SYSTEM_PROMPT = ROOT / "annotation" / "prompts" / "compact_system.md"

N_TRANSLATE = 100
SAMPLE_SIZE = 300
MAX_ATTEMPTS = 3
CONFIDENCES = {"high", "medium", "low"}
TAGS = {"sarcasm", "negation", "implicit", "code_mixed", "romanised", "question"}
LANG_CHECK = {"hi", "bn", "other"}
LABEL_FIX = {"favour": "Favour", "favor": "Favour", "against": "Against", "none": "None", "neutral": "None"}


@dataclass(frozen=True)
class Annotator:
    name: str
    provider: str
    model: str
    batch_size: int
    min_interval: float  # seconds between request starts
    extra: dict


ANNOTATORS = {
    "gemini": Annotator("gemini", "gemini", "gemini-3.5-flash-lite", 25, 4.2, {}),
    "gptoss": Annotator("gptoss", "groq", "openai/gpt-oss-120b", 50, 2.0, {"reasoning_effort": "medium"}),
}


class DailyLimitReached(RuntimeError):
    pass


class RequestTooLarge(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ============================================================================================ prepare


def build_main_items(clean: pd.DataFrame, removed: pd.DataFrame, min_tokens: int = 3) -> pd.DataFrame:
    """comments_clean rows (lang_uncertain=0) + other_language rows with >= min_tokens (lang_uncertain=1)."""
    main = clean[["id", "video_id", "text", "lang", "script"]].assign(lang_uncertain=0)
    unc = removed[(removed["reason"] == "other_language") & (removed["n_tokens"].astype(int) >= min_tokens)]
    unc = unc.assign(text=unc["text"].map(clean_text), lang="", lang_uncertain=1)
    unc = unc.assign(script=unc["text"].map(tag_script))
    items = pd.concat([main, unc[["id", "video_id", "text", "lang", "script", "lang_uncertain"]]], ignore_index=True)
    if items["id"].duplicated().any():
        raise ValueError("duplicate ids in labelling items")
    return items


def select_translation_subset(gold: pd.DataFrame, n: int = N_TRANSLATE, seed: int = SEED) -> pd.DataFrame:
    """Stratified by human label (proportional, at least 15 per class), seed 42."""
    counts = gold["label"].value_counts()
    quota = {lab: max(15, round(n * c / len(gold))) for lab, c in counts.items()}
    while sum(quota.values()) > n:  # trim the largest class
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(quota, key=quota.get)] += 1
    parts = [gold[gold["label"] == lab].sample(k, random_state=seed) for lab, k in quota.items()]
    sub = pd.concat(parts).sort_index()
    return pd.DataFrame({"gold_id": sub["id"], "label": sub["label"],
                         "mapping_confidence": sub["mapping_confidence"], "text_en": sub["text"],
                         "text_hi": "", "text_bn": ""})


def build_gold_items(gold: pd.DataFrame, translations: pd.DataFrame | None) -> pd.DataFrame:
    en = pd.DataFrame({"id": gold["id"], "gold_id": gold["id"], "set": "gold_en", "lang": "en", "text": gold["text"],
                       "human_label": gold["label"], "mapping_confidence": gold["mapping_confidence"]})
    parts = [en]
    if translations is not None:
        for lang in ("hi", "bn"):
            t = translations[translations[f"text_{lang}"].astype(str).str.strip() != ""]
            parts.append(pd.DataFrame({
                "id": t["gold_id"] + f"__{lang}", "gold_id": t["gold_id"], "set": f"gold_{lang}", "lang": lang,
                "text": t[f"text_{lang}"], "human_label": t["label"], "mapping_confidence": t["mapping_confidence"]}))
    return pd.concat(parts, ignore_index=True)


def check_translations(tr: pd.DataFrame) -> list[str]:
    """Problems with the translation file: empty cells or the wrong script."""
    problems = []
    for lang, script in (("hi", "devanagari"), ("bn", "bengali")):
        col = tr[f"text_{lang}"].fillna("").astype(str)
        for gid in tr.loc[col.str.strip() == "", "gold_id"]:
            problems.append(f"{gid}: text_{lang} empty")
        for gid, text in zip(tr["gold_id"], col):
            if text.strip() and tag_script(text) not in (script, "mixed"):
                problems.append(f"{gid}: text_{lang} script is {tag_script(text)}")
    return problems


def cmd_prepare(args) -> None:
    clean = pd.read_csv(CLEAN_CSV, dtype=str, keep_default_na=False)
    removed = pd.read_csv(REMOVED_CSV, dtype=str, keep_default_na=False)
    items = build_main_items(clean, removed)
    items.to_csv(ITEMS_CSV, index=False, encoding="utf-8")
    print(f"wrote {rel(ITEMS_CSV)}: {len(items)} items ({int(items['lang_uncertain'].sum())} lang_uncertain)")

    gold = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    if not TRANSLATIONS_CSV.exists():
        select_translation_subset(gold).to_csv(TRANSLATIONS_CSV, index=False, encoding="utf-8")
        print(f"wrote translation template {rel(TRANSLATIONS_CSV)} (fill text_hi and text_bn)")
    tr = pd.read_csv(TRANSLATIONS_CSV, dtype=str, keep_default_na=False)
    problems = check_translations(tr)
    print(f"translations: {len(tr)} rows, {len(problems)} problems" + (f" (first: {problems[:3]})" if problems else ""))
    gi = build_gold_items(gold, tr)
    gi.to_csv(GOLD_ITEMS_CSV, index=False, encoding="utf-8")
    print(f"wrote {rel(GOLD_ITEMS_CSV)}: {gi['set'].value_counts().to_dict()}")


# ============================================================================================ API calls


def load_env() -> dict:
    from dotenv import dotenv_values

    return dotenv_values(ROOT / ".env")


def user_message(batch: pd.DataFrame) -> tuple[str, dict]:
    """JSON list with short ids 1..n; returns (message, short id -> real id)."""
    payload, idmap = [], {}
    for i, row in enumerate(batch.itertuples(index=False), start=1):
        item = {"id": str(i), "text": row.text}
        if getattr(row, "lang_check", False):
            item["lang_check"] = True
        payload.append(item)
        idmap[str(i)] = row.id
    return json.dumps(payload, ensure_ascii=False), idmap


def call_gemini(ann: Annotator, key: str, system: str, message: str) -> tuple[str, dict]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ann.model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": message}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "maxOutputTokens": 8192}}
    r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=180)
    if r.status_code == 429:
        err = r.json().get("error", {})
        quota_ids = [v.get("quotaId", "") for d in err.get("details", []) for v in d.get("violations", [])]
        delay = next((d["retryDelay"] for d in err.get("details", []) if "retryDelay" in d), "10s")
        if any("PerDay" in q for q in quota_ids):
            raise DailyLimitReached(f"Gemini daily quota: {quota_ids}")
        raise RateLimited(float(str(delay).rstrip("s") or 10) + 1)
    if r.status_code >= 500:
        raise TransientError(f"HTTP {r.status_code}")
    if not r.ok:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    cand = (j.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    usage = j.get("usageMetadata", {})
    return text, {"in": usage.get("promptTokenCount"), "out": usage.get("candidatesTokenCount"),
                  "finish": cand.get("finishReason")}


def call_groq(ann: Annotator, key: str, system: str, message: str) -> tuple[str, dict]:
    body = {"model": ann.model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}], **ann.extra}
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=300)
    if r.status_code == 429:
        msg = r.text
        if re.search(r"per day|\(TPD\)|\(RPD\)", msg, re.IGNORECASE):
            raise DailyLimitReached(f"Groq daily limit: {msg[:200]}")
        raise RateLimited(float(r.headers.get("retry-after", 10)) + 1)
    if r.status_code == 413 or "too large" in r.text.lower():
        raise RequestTooLarge(r.text[:200])
    if r.status_code >= 500:
        raise TransientError(f"HTTP {r.status_code}")
    if not r.ok:
        raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    usage = j.get("usage", {})
    return j["choices"][0]["message"]["content"], {"in": usage.get("prompt_tokens"),
                                                   "out": usage.get("completion_tokens"),
                                                   "finish": j["choices"][0].get("finish_reason")}


class RateLimited(RuntimeError):
    def __init__(self, wait: float):
        super().__init__(f"rate limited, wait {wait:.0f}s")
        self.wait = wait


class TransientError(RuntimeError):
    pass


# ============================================================================================ validation


def validate_response(text: str, idmap: dict, lang_check_ids: set[str]) -> tuple[dict, dict]:
    """Parse a model response. Returns ({real_id: record}, {real_id: error}) covering every id in idmap."""
    valid, errors = {}, {}
    try:
        data = json.loads(text)
        entries = data["labels"] if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError("labels is not a list")
    except (ValueError, KeyError, TypeError) as err:
        return {}, {rid: f"unparseable response: {err}" for rid in idmap.values()}
    for e in entries:
        if not isinstance(e, dict) or str(e.get("id")) not in idmap:
            continue
        rid = idmap[str(e["id"])]
        if rid in valid:
            errors[rid] = "duplicate id in response"
            continue
        label = LABEL_FIX.get(str(e.get("label", "")).strip().lower())
        conf = str(e.get("confidence", "")).strip().lower()
        tags = e.get("tags", [])
        if label not in LABELS:
            errors[rid] = f"bad label {e.get('label')!r}"
            continue
        if conf not in CONFIDENCES:
            errors[rid] = f"bad confidence {e.get('confidence')!r}"
            continue
        if not isinstance(tags, list):
            tags = []
        rec = {"id": rid, "label": label, "confidence": conf, "tags": sorted({str(t) for t in tags} & TAGS)}
        if rid in lang_check_ids:
            lang = str(e.get("lang", "")).strip().lower()
            if lang not in LANG_CHECK:
                errors[rid] = f"bad lang {e.get('lang')!r}"
                continue
            rec["lang"] = lang
        valid[rid] = rec
    for rid in idmap.values():
        if rid not in valid and rid not in errors:
            errors[rid] = "missing from response"
    for rid in list(errors):
        if rid in valid:
            del valid[rid]
    return valid, errors


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a half-written last line from an interrupted run
    return out


def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path) if r.get("label") in LABELS}


# ============================================================================================ runner


def label_items(items: pd.DataFrame, ann: Annotator, out: Path, set_name: str, max_requests: int | None,
                key: str, call=None, sleep=time.sleep) -> dict:
    """Label ``items`` (id, text, lang_check) into ``out``; resumes, retries, logs. Returns run stats."""
    call = call or (call_gemini if ann.provider == "gemini" else call_groq)
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    fail_log = out.with_name(out.stem.replace("_gold", "") + "_failures.jsonl")
    req_log = out.with_name(out.stem.replace("_gold", "") + "_requests.jsonl")
    todo = items[~items["id"].isin(done_ids(out))].copy()
    attempts = {i: 0 for i in todo["id"]}
    queue = [todo.iloc[i:i + ann.batch_size] for i in range(0, len(todo), ann.batch_size)]
    stats = {"todo_at_start": len(todo), "labelled": 0, "failed": 0, "requests": 0, "stopped": ""}
    print(f"[{ann.name}/{set_name}] {len(items)} items, {len(items) - len(todo)} already done, "
          f"{len(todo)} to label in {len(queue)} batches")
    last_start, t0 = 0.0, time.time()
    while queue:
        if max_requests is not None and stats["requests"] >= max_requests:
            stats["stopped"] = f"max_requests={max_requests}"
            break
        batch = queue.pop(0)
        message, idmap = user_message(batch)
        lang_ids = set(batch.loc[batch["lang_check"].astype(bool), "id"])
        wait = ann.min_interval - (time.time() - last_start)
        if wait > 0:
            sleep(wait)
        last_start = time.time()
        stats["requests"] += 1
        try:
            text, usage = call(ann, key, system, message)
            status = "ok"
        except RateLimited as err:
            append_jsonl(req_log, [{"ts": now(), "set": set_name, "n": len(batch), "status": "rate_limited"}])
            queue.insert(0, batch)
            sleep(err.wait)
            continue
        except DailyLimitReached as err:
            queue.insert(0, batch)
            stats["stopped"] = str(err)
            break
        except RequestTooLarge:
            half = len(batch) // 2 or 1
            queue[:0] = [batch.iloc[:half], batch.iloc[half:]] if len(batch) > 1 else []
            append_jsonl(req_log, [{"ts": now(), "set": set_name, "n": len(batch), "status": "too_large_split"}])
            continue
        except (TransientError, requests.RequestException) as err:
            text, usage, status = "", {}, f"error: {err}"
        except RuntimeError as err:
            text, usage, status = "", {}, f"error: {err}"
        valid, errors = validate_response(text, idmap, lang_ids) if text else (
            {}, {rid: status for rid in idmap.values()})
        stamp = now()
        append_jsonl(out, [{**rec, "annotator": ann.name, "model": ann.model, "set": set_name, "ts": stamp}
                           for rec in valid.values()])
        append_jsonl(req_log, [{"ts": stamp, "set": set_name, "n": len(batch), "status": status,
                                "valid": len(valid), "invalid": len(errors), **usage}])
        stats["labelled"] += len(valid)
        retry = []
        for rid, err in errors.items():
            attempts[rid] += 1
            if attempts[rid] >= MAX_ATTEMPTS:
                append_jsonl(fail_log, [{"id": rid, "set": set_name, "error": err, "attempts": attempts[rid],
                                         "ts": stamp}])
                stats["failed"] += 1
            else:
                retry.append(rid)
        if retry:
            queue.append(batch[batch["id"].isin(retry)])
        done = len(items) - len(todo) + stats["labelled"]
        rate = stats["labelled"] / max(time.time() - t0, 1e-9) * 60
        eta = (len(items) - done) / rate if rate else float("nan")
        print(f"  req {stats['requests']:>4}: +{len(valid)}/{len(batch)} valid; {done}/{len(items)} done; "
              f"{rate:.0f} items/min; ETA {eta:.0f} min" + (f"; {len(errors)} invalid" if errors else ""),
              flush=True)
    stats["remaining"] = len(items) - len(done_ids(out) & set(items["id"]))
    print(f"[{ann.name}/{set_name}] stopped: {stats['stopped'] or 'all done'}; {stats}")
    return stats


def load_items(set_name: str) -> pd.DataFrame:
    if set_name == "main":
        df = pd.read_csv(ITEMS_CSV, dtype=str, keep_default_na=False)
        return df.assign(lang_check=df["lang_uncertain"] == "1")
    df = pd.read_csv(GOLD_ITEMS_CSV, dtype=str, keep_default_na=False)
    if set_name == "gold_en":
        df = df[df["set"] == "gold_en"]
    return df.assign(lang_check=False)


def out_path(annotator: str, set_name: str, dry: bool = False) -> Path:
    suffix = "_gold" if set_name.startswith("gold") else ""
    return INTERIM / f"labels_{annotator}{suffix}{'.dryrun' if dry else ''}.jsonl"


def cmd_run(args) -> None:
    ann = ANNOTATORS[args.annotator]
    items = load_items(args.set)
    if args.limit:
        items = items.head(args.limit)
    out = out_path(ann.name, args.set, bool(args.limit))
    key = load_env()["GEMINI_API_KEY" if ann.provider == "gemini" else "GROQ_API_KEY"]
    label_items(items, ann, out, args.set, args.max_requests, key)


# ============================================================================================ adjudication


def labels_frame(path: Path) -> pd.DataFrame:
    recs = [r for r in read_jsonl(path) if r.get("label") in LABELS]
    df = pd.DataFrame(recs, columns=["id", "label"] + (["lang"] if any("lang" in r for r in recs) else []))
    return df.drop_duplicates("id", keep="last")


def adjudication_queue(items: pd.DataFrame, claude: pd.DataFrame, gemini: pd.DataFrame, gold_en: pd.DataFrame,
                       sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> list[tuple[str, pd.DataFrame]]:
    """[(set name, items)] in priority order: disagreements, English gold, random sample of the rest."""
    missing = set(items["id"]) - (set(claude["id"]) & set(gemini["id"]))
    if missing:
        raise SystemExit(f"{len(missing)} items lack a Claude or Gemini label; finish both first "
                         f"(e.g. {sorted(missing)[:3]})")
    both = claude.merge(gemini, on="id", suffixes=("_claude", "_gemini"))
    disagree_ids = set(both.loc[both["label_claude"] != both["label_gemini"], "id"])
    disagree = items[items["id"].isin(disagree_ids)]
    rest = items[~items["id"].isin(disagree_ids)]
    sample = rest.sample(min(sample_size, len(rest)), random_state=seed)
    return [("disagreement", disagree), ("gold_en", gold_en), ("sample", sample)]


def cmd_adjudicate(args) -> None:
    ann = ANNOTATORS["gptoss"]
    items = load_items("main")
    queue = adjudication_queue(items, labels_frame(out_path("claude", "main")), labels_frame(out_path("gemini", "main")),
                               load_items("gold_en"))
    key = load_env()["GROQ_API_KEY"]
    budget = args.max_requests
    for set_name, part in queue:
        stats = label_items(part, ann, out_path("gptoss", set_name), set_name, budget, key)
        if budget is not None:
            budget -= stats["requests"]
        if stats["stopped"] and not stats["stopped"].startswith("all"):
            print(f"Stopped during {set_name}: {stats['stopped']}. Re-run tomorrow to resume.")
            return


# ============================================================================================ validate


def validate_file(path: Path, expected_ids: set[str], lang_check_ids: set[str], need_gloss: bool = False) -> dict:
    recs = read_jsonl(path)
    ids = [r.get("id") for r in recs]
    bad = []
    for r in recs:
        if r.get("label") not in LABELS or str(r.get("confidence", "")).lower() not in CONFIDENCES:
            bad.append(r.get("id"))
        elif r.get("id") in lang_check_ids and r.get("lang") not in LANG_CHECK:
            bad.append(r.get("id"))
        elif need_gloss and not (r.get("gloss") and r.get("reason")):
            bad.append(r.get("id"))
    got = {r["id"] for r in recs if r.get("id") not in bad}
    return {"file": rel(path), "lines": len(recs), "duplicate_ids": len(ids) - len(set(ids)),
            "invalid": len(bad), "unknown_ids": len(got - expected_ids),
            "missing": len(expected_ids - got), "missing_examples": sorted(expected_ids - got)[:5]}


def cmd_validate(args) -> None:
    items, gold = load_items("main"), load_items("gold")
    need = args.annotator == "claude"
    for set_name, frame in (("main", items), ("gold", gold)):
        path = out_path(args.annotator, set_name)
        report = validate_file(path, set(frame["id"]), set(frame.loc[frame["lang_check"], "id"]), need)
        print(json.dumps(report, ensure_ascii=False))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="write labelling_items.csv, gold_items.csv and the translation template")
    r = sub.add_parser("run", help="label a set with an API annotator")
    r.add_argument("--annotator", choices=sorted(ANNOTATORS), required=True)
    r.add_argument("--set", choices=["gold", "gold_en", "main"], required=True)
    r.add_argument("--max-requests", type=int, default=None)
    r.add_argument("--limit", type=int, default=None, help="dry run on the first N items (*.dryrun.jsonl)")
    a = sub.add_parser("adjudicate", help="gpt-oss on disagreements, English gold, then a 300 sample")
    a.add_argument("--max-requests", type=int, default=None)
    v = sub.add_parser("validate", help="check an annotator's label files")
    v.add_argument("--annotator", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    {"prepare": cmd_prepare, "run": cmd_run, "adjudicate": cmd_adjudicate, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
