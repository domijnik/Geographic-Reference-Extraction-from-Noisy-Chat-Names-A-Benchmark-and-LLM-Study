"""
Parse results.jsonl from run_experiment.py into tables.

pip install pandas openpyxl
python parse_results.py                 # reads results.jsonl
python parse_results.py my_results.jsonl

Writes:
  parsed_results.csv   one row per chat x model x prompt (list fields joined with " | ")
  parse_summary.csv    per model x prompt: how many chats parsed, missing, invalid
  parsed_results.xlsx  both tables as sheets (if openpyxl is installed)
"""
import json, re, sys
import pandas as pd

IN = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"

FIELDS = ["Contains_Geo_Reference", "Reference_Original", "Reference_Country",
          "Reference_Region", "Reference_City", "Reference_Type",
          "Confidence", "Comment"]
LIST_FIELDS = ["Reference_Original", "Reference_Country",
               "Reference_Region", "Reference_City"]
ALLOWED = {
    "Contains_Geo_Reference": {"Ja", "Nein", "Unklar"},
    "Reference_Type": {"Text", "Abbreviation", "Postal Code", "Emoji",
                       "Multiple", "Undefined", None},
    "Confidence": {"Unambiguous", "Ambiguous"},
}


def extract_json(text):
    """Pull the JSON array out of a model reply, tolerating common noise."""
    if text is None:
        raise ValueError("empty reply")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # Qwen thinking
    text = re.sub(r"```(?:json)?", "", text).strip()            # code fences
    start = min((i for i in (text.find("["), text.find("{")) if i != -1),
                default=-1)
    if start == -1:
        raise ValueError("no JSON found")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if isinstance(obj, dict):          # a single object instead of an array
        obj = [obj]
    if not isinstance(obj, list):
        raise ValueError("JSON is not an array")
    return obj


def fix_list(v):
    """Normalise list fields: [] / "" / "null" -> None, a bare string -> [string]."""
    if v in (None, "", [], "null", "None"):
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        v = [str(x) for x in v if x not in (None, "")]
        return v or None
    return [str(v)]


def problems(item):
    errs = [f"missing {f}" for f in FIELDS if f not in item]
    for f, allowed in ALLOWED.items():
        if f in item and item[f] not in allowed:
            errs.append(f"{f}={item[f]!r} not allowed")
    return errs


# Read records. A batch may appear more than once (an error, then a successful
# retry on rerun); keep the last successful copy, else the last error.
batches = {}
with open(IN, encoding="utf-8") as fh:
    for line_no, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"skipping corrupt line {line_no}")
            continue
        is_err = str(rec.get("output", "")).startswith("__ERROR__")
        prev = batches.get(rec["key"])
        if prev is None or not is_err or prev["_err"]:
            rec["_err"] = is_err
            batches[rec["key"]] = rec

rows = []
for rec in batches.values():
    inputs = rec.get("input") or [{"Chat_ID": None, "Chat_Name_Raw": n}
                                  for n in rec.get("names", [])]
    base = {"model": rec["model"], "prompt": rec["prompt"]}

    if rec["_err"]:
        for c in inputs:
            rows.append({**base, **c, "status": "api_error",
                         "issues": rec["output"][:300]})
        continue
    try:
        items = extract_json(rec["output"])
    except (ValueError, json.JSONDecodeError) as e:
        for c in inputs:
            rows.append({**base, **c, "status": "unparseable",
                         "issues": str(e), "raw_output": rec["output"][:2000]})
        continue

    # Match answers to inputs by Chat_ID; fall back to position if IDs are off.
    by_id = {str(it.get("Chat_ID")): it for it in items if isinstance(it, dict)}
    for pos, c in enumerate(inputs):
        it = by_id.get(str(c["Chat_ID"]))
        how = "id"
        if it is None and pos < len(items) and isinstance(items[pos], dict) \
                and len(items) == len(inputs):
            it, how = items[pos], "position"
        if it is None:
            rows.append({**base, **c, "status": "missing",
                         "issues": "no answer for this chat"})
            continue
        issues = problems(it)
        if how == "position":
            issues.append("matched by position (Chat_ID differed)")
        if it.get("Chat_Name_Raw") not in (None, c["Chat_Name_Raw"]):
            issues.append("Chat_Name_Raw changed by model")
        row = {**base, **c, "status": "ok" if not problems(it) else "invalid",
               "issues": "; ".join(issues)}
        for f in FIELDS:
            v = it.get(f)
            if f in LIST_FIELDS:
                v = fix_list(v)
                v = " | ".join(v) if v else None
            row[f] = v
        rows.append(row)

df = pd.DataFrame(rows)
cols = ["model", "prompt", "Chat_ID", "Chat_Name_Raw", "status", *FIELDS,
        "issues", "raw_output"]
df = df.reindex(columns=cols).sort_values(["model", "prompt", "Chat_ID"],
                                           kind="stable")

summary = (df.groupby(["model", "prompt"])["status"]
             .value_counts().unstack(fill_value=0))
summary["total"] = summary.sum(axis=1)
if "ok" in summary:
    summary["ok_pct"] = (100 * summary["ok"] / summary["total"]).round(1)

df.to_csv("parsed_results.csv", index=False, encoding="utf-8-sig")
summary.to_csv("parse_summary.csv", encoding="utf-8-sig")
try:
    with pd.ExcelWriter("parsed_results.xlsx") as xw:
        df.to_excel(xw, sheet_name="results", index=False)
        summary.to_excel(xw, sheet_name="summary")
except ImportError:
    pass

print(summary.to_string())
print(f"\n{len(df)} rows -> parsed_results.csv, parse_summary.csv")