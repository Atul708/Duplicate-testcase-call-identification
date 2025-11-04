#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import math
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime

# ======================================================
# STEP 0: ASK USER FOR INPUT JSON
# ======================================================
print("📂 Please provide the full path of your JSON input file (e.g. D:\\Office Data\\Downloads\\bcom__ios___.json):")
path_input = input("➡️ Enter JSON file path: ").strip().strip('"')
path = Path(path_input)
if not path.exists():
    raise FileNotFoundError(f"❌ File not found: {path}")
print(f"\n✅ Input file found: {path}")

# ======================================================
# STEP 1: JSON → CSV PARSING WITH TIMESTAMP
# ======================================================
TESTCASE_DELIMS = ["]}}]}", "]}}],", "]}}]},", "]}},"]

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

chosen_delim = next((d for d in TESTCASE_DELIMS if d in text), TESTCASE_DELIMS[0])
chunks = text.split(chosen_delim)

rows, case_num = [], 0
re_evar = re.compile(r'"(eVar\d+)"\s*:\s*"([^"]*)"')
re_prop = re.compile(r'"(prop\d+)"\s*:\s*"([^"]*)"')
re_events = re.compile(r'"events"\s*:\s*"([^"]*)"')
re_product_string = re.compile(r'"product-string"\s*:\s*"([^"]*)"')
re_track_action = re.compile(r'"trackAction"\s*:\s*"([^"]*)"|"trackaction"\s*:\s*"([^"]*)"')
re_msg_time = re.compile(r'"messageReceivedTime"\s*:\s*\{\s*"date"\s*:\s*"([^"]*)"')
re_impression_data = re.compile(r'"impressionData"\s*:\s*"([^"]*)"')
re_custom_event = re.compile(r'"custom_event_name"\s*:\s*"([^"]*)"')
re_ts_ms = re.compile(r'"timestamp"\s*:\s*(\d{10,}(?:\.\d+)?)')

for chunk in chunks:
    if not chunk.strip():
        continue
    case_num += 1
    row = {"test_case_id": case_num}

    for k, v in re_evar.findall(chunk):
        row[k] = v
    for k, v in re_prop.findall(chunk):
        row[k] = v

    for pat, name in [
        (re_events, "events"),
        (re_product_string, "product-string"),
        (re_impression_data, "impressionData"),
        (re_custom_event, "custom_event_name"),
        (re_msg_time, "messageReceivedTime")
    ]:
        m = pat.search(chunk)
        if m:
            row[name] = m.group(1)

    m = re_track_action.search(chunk)
    if m:
        row["trackaction"] = m.group(1) or m.group(2)

    m = re_ts_ms.search(chunk)
    if m:
        row["timestamp"] = m.group(1)

    rows.append(row)

df = pd.DataFrame(rows)

# Shift timestamp up by one row
if "timestamp" in df.columns:
    df["timestamp"] = df["timestamp"].shift(-1)
    df.loc[df.index[-1], "timestamp"] = pd.NA

# Add IST timestamp
if "timestamp" in df.columns:
    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    ts_num = ts_num.apply(lambda x: math.ceil(x) + 1 if pd.notna(x) else pd.NA)
    ts_utc = pd.to_datetime(ts_num, unit="ms", errors="coerce", utc=True)
    ts_ist = ts_utc.dt.tz_convert("Asia/Kolkata")
    df["TIMESTAMP"] = "'" + ts_ist.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
else:
    df["TIMESTAMP"] = pd.NA

out_step1 = path.with_name(f"{path.stem}_parsed.csv")
df.to_csv(out_step1, index=False, encoding="utf-8")
print(f"✅ Step 1: Parsed CSV saved → {out_step1}")

# ======================================================
# STEP 2: MODULE IDENTIFICATION (YOUR CODE)
# ======================================================
INPUT_PATH = out_step1
OUT_PATH = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_modules.csv")

df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False).replace({"": pd.NA})
original_order = list(df.columns)

# =========================
# MODULE KEYWORDS & PRIORITY
# =========================
MODULE_KEYWORDS = {
    "Home page":   {"custom": ["onboarding", "home page", "notification response", "content link click"],
                    "action": ["onboarding", "home page", "notification response"]},
    "Search":      {"custom": ["search"], "action": ["search"]},
    "Navigation":  {"custom": ["nav", "navigation"], "action": ["nav", "navigation"]},
    "My account":  {"custom": ["sign-in", "my account", "login", "account sign-in", "store"],
                    "action": ["sign-in", "my account", "login", "account sign-in", "store"]},
    "Wishlist":    {"custom": ["wishlist"], "action": ["wishlist"]},
    "PDP":         {"custom": ["pdp", "product", "product click"],
                    "action": ["pdp", "product", "product click"]},
    "Browse":      {"custom": ["browse"], "action": ["browse"]},
    "Pros":        {"custom": ["pros"], "action": ["pros"]},
    "Omni":        {"custom": ["omni"], "action": ["omni"]},
    "Registry":    {"custom": ["registry"], "action": ["registry"]},
    "PPE Return":  {"custom": ["ppe return"], "action": ["ppe return"]},
    "Bag":         {"custom": ["bag", "add to bag"], "action": ["bag", "add to bag"]},
    "Checkout":    {"custom": ["checkout", "step up login", "login & checkout", "shipping", "contact info", "order confirmation", "payment"],
                    "action": ["checkout", "step up login", "login & checkout", "payment"]},
    "App Specific":{"custom": ["app specific", "app-specific"], "action": ["app specific", "app-specific"]},
}

PRIORITY_COLS = ["eVar11","eVar164","eVar200","eVar216","eVar45","eVar43","eVar3","prop2"]

def _norm(s): return "" if pd.isna(s) else str(s).strip().lower()
def modules_in_text(text): return {mod for mod, kw in MODULE_KEYWORDS.items() if any(w in _norm(text) for w in kw["custom"] + kw["action"])}
def text_modules(row):
    ce = row.get("custom_event_name", "")
    ta = row.get("trackaction", row.get("trackAction", ""))
    if "tab bar bottom nav" in _norm(ta):
        return ["Navigation"]
    hits = modules_in_text(ce) | modules_in_text(ta)
    return sorted(hits)
def first_priority_modules(row):
    for col in PRIORITY_COLS:
        if col in row:
            mods = sorted(modules_in_text(row[col]))
            if mods:
                return mods
    return []
def e43_45_single(row):
    mods = modules_in_text(row.get("eVar43", "")) | modules_in_text(row.get("eVar45", ""))
    return list(mods) if len(mods) == 1 else []

# =========================
# MODULE LOGIC
# =========================
initial_modules = []
for _, r in df.iterrows():
    m = text_modules(r)
    if not m: m = first_priority_modules(r)
    initial_modules.append(", ".join(m) if m else "")
df["module"] = initial_modules

# High-priority override
for i, r in df.iterrows():
    p43, p45 = _norm(r.get("prop43", "")), _norm(r.get("prop45", ""))
    if "account created" in p43 or "|account created" in p45: df.at[i, "module"] = "My account"

# Override multiples if eVar43/45 gives a single
for i, r in df.iterrows():
    cur = df.at[i, "module"] or ""
    if "," in cur or cur == "":
        single = e43_45_single(r)
        if single: df.at[i, "module"] = single[0]

# ZONE rule
for i, r in df.iterrows():
    if df.at[i, "module"] in ("", "Unknown"):
        e216 = _norm(r.get("eVar216", ""))
        if "sb_zone" in e216: df.at[i, "module"] = "Bag"
        elif "hp_zone" in e216: df.at[i, "module"] = "Home page"
df["module"] = df["module"].replace({"": "Unknown"})

# Next-row fill
nav_for_next = df.apply(lambda r: ", ".join(sorted(modules_in_text(r.get("eVar29", "")) | modules_in_text(r.get("eVar30", "")))) or "", axis=1)
df["_nav_for_next_str"] = nav_for_next
df["_nav_for_me"] = df["_nav_for_next_str"].shift(1)
mask_fill = (df["module"] == "Unknown") & df["_nav_for_me"].fillna("").ne("")
df.loc[mask_fill, "module"] = df.loc[mask_fill, "_nav_for_me"]

# Possible module & chain
def window_modules(idx, k=5): return [df.at[j, "module"] for j in range(max(0, idx-k), min(len(df), idx+k+1)) if j != idx]
def most_freq(mod_list):
    clean = [m for m in mod_list if m and m != "Unknown"]
    if not clean: return ""
    c = Counter(clean)
    m = max(c.values())
    return ", ".join(sorted([k for k, v in c.items() if v == m]))
def chain_view(idx, k=5):
    n = len(df)
    next_idxs = list(range(idx+1, min(n, idx+1+k)))
    next_part = [str(df.at[j, "module"]) for j in reversed(next_idxs)]
    prev_part = [str(df.at[j, "module"]) for j in range(idx-1, max(-1, idx-1-k), -1)]
    return " -> ".join(next_part + ["<<NO MODULE>>"] + prev_part)

df["possible_module"], df["module_chain"] = "", ""
unknown_mask = df["module"].eq("Unknown")
for i in df.index[unknown_mask]:
    prev_mod = df.at[i-1, "module"] if i-1 >=0 else ""
    next_mod = df.at[i+1, "module"] if i+1 < len(df) else ""
    if prev_mod and next_mod and prev_mod != "Unknown" and next_mod != "Unknown" and prev_mod == next_mod:
        df.at[i, "possible_module"] = prev_mod
    else:
        df.at[i, "possible_module"] = most_freq(window_modules(i))
    df.at[i, "module_chain"] = chain_view(i)

# =========================
# SAVE FINAL CSV
# =========================
def safe_to_csv(df, path, **kwargs):
    for i in range(6):
        p = path if i==0 else path.with_name(f"{path.stem}_unlocked{i}{path.suffix}")
        try: 
            df.to_csv(p, **kwargs)
            print(f"✅ Saved classified modules to: {p}")
            return p
        except PermissionError: continue
    raise PermissionError("Could not save file (locked).")

# =========================
# FINAL COLUMN ORDER
# =========================
def insert_after(lst, anchor, item):
    lst = [c for c in lst if c != item]
    if anchor in lst:
        idx = lst.index(anchor)+1
        return lst[:idx] + [item] + lst[idx:]
    return lst + [item]

final_cols = original_order.copy()
for c in ["module", "possible_module", "module_chain", "_nav_for_next_str"]:
    if c not in df.columns: df[c] = pd.NA

final_cols = insert_after(final_cols, "custom_event_name", "module")
for tail_col in ["possible_module", "module_chain", "_nav_for_next_str"]:
    if tail_col in final_cols: final_cols.remove(tail_col)
final_cols += ["possible_module", "module_chain", "_nav_for_next_str"]
for c in df.columns:
    if c not in final_cols: final_cols.append(c)

df = df.reindex(columns=final_cols)
safe_to_csv(df, OUT_PATH, index=False, encoding="utf-8")

