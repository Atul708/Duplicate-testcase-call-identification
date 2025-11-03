#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pipeline_gdrive_full_v2.py
# Accepts a Google Drive share link OR a Drive/local path.
# Robust to files with NO extension (gdown sometimes saves as "downloaded_file"):
#   - Detects JSON/CSV/XLSX by content and renames with the right extension.
# Full pipeline:
#   1) JSON-ish -> CSV (if input is JSON)
#   2) Module identification
#   3) Adjacent duplicate-call identification (30s window)
#   4) Collapse duplicate groups to one representative (unique rows)
# Prints ONLY the final table in a nice box and saves outputs next to the input.

from __future__ import annotations
import re
import sys
import os
import io
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Tuple
import pandas as pd

# ===================== Pretty print in a box =====================
try:
    from tabulate import tabulate
    def pretty_print(df: pd.DataFrame, limit: int = 200):
        print(tabulate(df.head(limit), headers="keys", tablefmt="fancy_grid", showindex=False))
except Exception:
    def pretty_print(df: pd.DataFrame, limit: int = 200):
        box = df.head(limit).to_string(index=False)
        line = "=" * max(80, len(box.splitlines()[0]) if box else 80)
        print(line); print(box); print(line)

# ===================== Colab / Drive helpers =====================
def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def ensure_colab_drive_mounted() -> None:
    if not in_colab():
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass

def resolve_gdrive_path(user_text: str) -> Path:
    s = user_text.strip().strip('"').strip("'")
    p = Path(s)
    if in_colab():
        ensure_colab_drive_mounted()
        if not p.is_absolute():
            p = Path("/content/drive/MyDrive") / s
    return p

def have_gdown() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except Exception:
        return False

class _ChDir:
    def __init__(self, path: Path): self.path = path; self._old = None
    def __enter__(self):
        self._old = Path.cwd()
        os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)
    def __exit__(self, exc_type, exc, tb):
        if self._old: os.chdir(self._old)

def download_from_drive_url(url: str) -> Path:
    """
    Download a share link via gdown and return the actual file path.
    Attempts to preserve the Drive filename (including extension).
    """
    # Extract file id
    m = re.search(r"/d/([^/]+)/", url) or re.search(r"[?&]id=([^&]+)", url)
    if not m:
        raise SystemExit("ERROR: Could not extract file id from the Drive link.")

    file_id = m.group(1)

    if not have_gdown():
        print("Installing gdown ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])

    import gdown

    # Choose a download directory
    if in_colab():
        ensure_colab_drive_mounted()
        dl_dir = Path("/content/drive/MyDrive/_pipeline_inputs")
    else:
        dl_dir = Path.home() / "Downloads" / "_pipeline_inputs"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Change CWD so gdown saves here and preserves filename
    with _ChDir(dl_dir):
        out = gdown.download(id=file_id, output=None, quiet=False, fuzzy=True)
        if not out:
            raise SystemExit("ERROR: Download failed. Ensure link is public or you are authenticated.")
        p = (dl_dir / out).resolve()
        if not p.exists():
            # Some gdown versions name it generically
            cand = dl_dir / "downloaded_file"
            if cand.exists():
                p = cand
            else:
                raise SystemExit("ERROR: Download reported success but file not found.")
    print(f"✅ Download complete! Saved to: {p}")
    return p

def resolve_input_any(user_text: str) -> Path:
    s = user_text.strip().strip('"').strip("'")
    if "drive.google.com" in s:
        return download_from_drive_url(s)
    return resolve_gdrive_path(s)

# ===================== File-type detection & normalization =====================
def sniff_filetype_and_fix_extension(path: Path) -> Path:
    """
    If the file has no extension, try to detect JSON/CSV/XLSX:
      - XLSX if starts with ZIP magic 'PK\x03\x04'
      - JSON if first non-whitespace char is '{' or '['
      - else treat as CSV
    Then rename the file to include the inferred extension.
    """
    if path.suffix:
        return path  # already has extension

    # Read a bit to sniff
    with open(path, "rb") as f:
        head = f.read(4096)
    # XLSX?
    if head[:2] == b"PK":
        newp = path.with_suffix(".xlsx")
        path.rename(newp)
        print(f"ℹ️ No extension detected; inferred XLSX and renamed to: {newp.name}")
        return newp

    # Try text sniff
    try:
        text = head.decode("utf-8", errors="ignore").lstrip()
    except Exception:
        text = ""

    # JSON?
    if text.startswith("{") or text.startswith("["):
        newp = path.with_suffix(".json")
        path.rename(newp)
        print(f"ℹ️ No extension detected; inferred JSON and renamed to: {newp.name}")
        return newp

    # Else assume CSV
    newp = path.with_suffix(".csv")
    path.rename(newp)
    print(f"ℹ️ No extension detected; inferred CSV and renamed to: {newp.name}")
    return newp

# ===================== Shared helpers =====================
def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    candidates = [path] + [path.with_name(f"{path.stem}_unlocked{i}{path.suffix}") for i in range(1, 6)]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates.append(path.with_name(f"{path.stem}_{ts}{path.suffix}"))
    last_err = None
    for p in candidates:
        try:
            df.to_csv(p, index=False, **kwargs)
            return p
        except PermissionError as e:
            last_err = e
            continue
    raise PermissionError(f"Could not write CSV (files may be locked). Last error: {last_err}")

def coerce_to_ts(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.notna().any(): return s
    try:
        num = pd.to_numeric(series, errors="coerce")
        s = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        if s.notna().any(): return s
        s = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
        return s
    except Exception:
        return pd.to_datetime(pd.Series([pd.NA] * len(series)), errors="coerce", utc=True)

def norm(v):
    if pd.isna(v): return pd.NA
    s = str(v).strip()
    return pd.NA if s == "" else s

def first_nonempty(row: pd.Series, keys: List[str]):
    for k in keys:
        if k in row:
            val = norm(row[k])
            if not pd.isna(val):
                return val
    return pd.NA

def safe_eq(a, b) -> bool:
    if pd.isna(a) and pd.isna(b): return True
    if pd.isna(a) or pd.isna(b): return False
    return a == b

# ===================== STEP 1: JSON-ish -> CSV =====================
TESTCASE_DELIMS = ["]}}]}", "]}}],", "]}}]},", "]}},"]

def parse_jsonish_to_csv(json_path: Path) -> Path:
    out_path = json_path.with_suffix(".csv")
    text = json_path.read_text(encoding="utf-8", errors="ignore")
    delim = next((d for d in TESTCASE_DELIMS if d in text), TESTCASE_DELIMS[0])
    chunks = text.split(delim)

    re_evar            = re.compile(r'"(eVar\d+)"\s*:\s*"([^"]*)"')
    re_prop            = re.compile(r'"(prop\d+)"\s*:\s*"([^"]*)"')
    re_events          = re.compile(r'"events"\s*:\s*"([^"]*)"')
    re_product_string  = re.compile(r'"product-string"\s*:\s*"([^"]*)"')
    re_track_action    = re.compile(r'"trackAction"\s*:\s*"([^"]*)"|"trackaction"\s*:\s*"([^"]*)"')
    re_msg_time        = re.compile(r'"messageReceivedTime"\s*:\s*\{\s*"date"\s*:\s*"([^"]*)"')
    re_impression_data = re.compile(r'"impressionData"\s*:\s*"([^"]*)"')
    re_custom_event    = re.compile(r'"custom_event_name"\s*:\s*"([^"]*)"')

    rows = []
    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip(): continue
        row = {"test_case_id": i}
        for k, v in re_evar.findall(chunk): row[k] = v
        for k, v in re_prop.findall(chunk): row[k] = v
        m = re_events.search(chunk);           row["events"] = m.group(1) if m else row.get("events")
        m = re_product_string.search(chunk);   row["product-string"] = m.group(1) if m else row.get("product-string")
        m = re_track_action.search(chunk);     row["trackaction"] = (m.group(1) or m.group(2)) if m else row.get("trackaction")
        m = re_custom_event.search(chunk);     row["custom_event_name"] = m.group(1) if m else row.get("custom_event_name")
        m = re_impression_data.search(chunk);  row["impressionData"] = m.group(1) if m else row.get("impressionData")
        m = re_msg_time.search(chunk);         row["messageReceivedTime"] = m.group(1) if m else row.get("messageReceivedTime")
        rows.append(row)

    df = pd.DataFrame(rows)

    def sort_cols(cols: List[str]) -> List[str]:
        lead = ["test_case_id","messageReceivedTime","trackaction","custom_event_name","events","product-string","impressionData"]
        ev = sorted([c for c in cols if re.fullmatch(r"eVar\d+", c, flags=re.I)])
        pr = sorted([c for c in cols if re.fullmatch(r"prop\d+", c, flags=re.I)])
        rest = [c for c in cols if c not in set(lead + ev + pr)]
        return [c for c in lead if c in cols] + ev + pr + rest

    if not df.empty:
        df = df.reindex(columns=sort_cols(list(df.columns)))

    return safe_to_csv(df, out_path, encoding="utf-8", errors="strict")

# ===================== STEP 1b: Excel (.xlsx) -> CSV =====================
def excel_to_csv(xlsx_path: Path) -> Path:
    df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str)
    df = df.replace({"": pd.NA})
    out = xlsx_path.with_suffix(".csv")
    return safe_to_csv(df, out, encoding="utf-8")

# ===================== STEP 2: Module Identification =====================
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
PRIORITY_COLS = ["eVar11","eVar164","eVar200","eVar216","eVar45","eVar43","eVar216","eVar164","eVar3","prop2"]

def _norm_text(s):
    return "" if pd.isna(s) else str(s).strip().lower()

def modules_in_text(text: str) -> set:
    t = _norm_text(text)
    if not t: return set()
    hits = set()
    for mod, kw in MODULE_KEYWORDS.items():
        if any(w and w in t for w in kw.get("custom", [])) or any(w and w in t for w in kw.get("action", [])):
            hits.add(mod)
    return hits

def text_modules(row) -> list:
    ce = row.get("custom_event_name", "")
    ta = row.get("trackaction", row.get("trackAction", ""))
    if "tab bar bottom nav" in _norm_text(ta):
        return ["Navigation"]
    return sorted(modules_in_text(ce) | modules_in_text(ta))

def first_priority_modules(row) -> list:
    for col in PRIORITY_COLS:
        if col in row:
            mods = sorted(modules_in_text(row[col]))
            if mods: return mods
    return []

def e43_45_single(row) -> list:
    mods = modules_in_text(row.get("eVar43", "")) | modules_in_text(row.get("eVar45", ""))
    return list(mods) if len(mods) == 1 else []

def classify_modules(input_csv_path: Path) -> Path:
    out_path = input_csv_path.with_name(f"{input_csv_path.stem}_modules.csv")
    df = pd.read_csv(input_csv_path, dtype=str, keep_default_na=False).replace({"": pd.NA})

    if "__ts" in df.columns:
        try: df["__ts"] = pd.to_datetime(df["__ts"], errors="coerce", utc=True)
        except Exception: pass

    initial = []
    for _, r in df.iterrows():
        m = text_modules(r)
        if not m: m = first_priority_modules(r)
        initial.append(", ".join(m) if m else "")
    df["module"] = initial

    # account created -> My account
    for i, r in df.iterrows():
        p43 = _norm_text(r.get("prop43", ""))
        p45 = _norm_text(r.get("prop45", ""))
        if "account created" in p43 or "|account created" in p45:
            df.at[i, "module"] = "My account"

    # eVar43/45 single override
    for i, r in df.iterrows():
        cur = df.at[i, "module"] or ""
        if "," in cur or cur == "":
            single = e43_45_single(r)
            if single: df.at[i, "module"] = single[0]

    # SB_ZONE / HP_ZONE in eVar216 if still Unknown/blank
    for i, r in df.iterrows():
        if df.at[i, "module"] in ("", "Unknown"):
            e216 = _norm_text(r.get("eVar216", ""))
            if "sb_zone" in e216: df.at[i, "module"] = "Bag"
            elif "hp_zone" in e216: df.at[i, "module"] = "Home page"

    df["module"] = df["module"].replace({"": "Unknown"})

    # eVar29/30 suggests next row
    nav_for_next = df.apply(lambda r: ", ".join(sorted(
        (modules_in_text(r.get("eVar29", "")) | modules_in_text(r.get("eVar30", "")))
    )) or "", axis=1)
    df["_nav_for_next_str"] = nav_for_next
    df["_nav_for_me"] = df["_nav_for_next_str"].shift(1)
    mask_fill = (df["module"] == "Unknown") & df["_nav_for_me"].fillna("").ne("")
    df.loc[mask_fill, "module"] = df.loc[mask_fill, "_nav_for_me"]

    # possible_module + module_chain for Unknowns
    def window_modules(idx: int, k: int = 5) -> list:
        lo = max(0, idx - k); hi = min(len(df), idx + k + 1)
        return [df.at[j, "module"] for j in range(lo, hi) if j != idx]

    def most_frequent_non_unknown(mod_list: list) -> str:
        clean = [m for m in mod_list if m and m != "Unknown"]
        if not clean: return ""
        counts = Counter(clean)
        max_cnt = max(counts.values())
        winners = sorted([m for m, c in counts.items() if c == max_cnt])
        return ", ".join(winners)

    def chain_view_next_then_prev(idx: int, k: int = 5) -> str:
        n = len(df)
        next_idxs = list(range(idx + 1, min(n, idx + 1 + k)))
        next_part = [str(df.at[j, "module"]) for j in reversed(next_idxs)]
        prev_part = [str(df.at[j, "module"]) for j in range(idx - 1, max(-1, idx - 1 - k), -1)]
        return " -> ".join(next_part + ["<<NO MODULE>>"] + prev_part)

    df["possible_module"] = ""
    df["module_chain"] = ""
    unknown_mask = df["module"].eq("Unknown")
    if unknown_mask.any():
        for i in df.index[unknown_mask]:
            prev_mod = df.at[i-1, "module"] if i-1 >= 0 else ""
            next_mod = df.at[i+1, "module"] if i+1 < len(df) else ""
            if prev_mod and next_mod and prev_mod != "Unknown" and next_mod != "Unknown" and prev_mod == next_mod:
                df.at[i, "possible_module"] = prev_mod
            else:
                df.at[i, "possible_module"] = most_frequent_non_unknown(window_modules(i, 5))
            df.at[i, "module_chain"] = chain_view_next_then_prev(i, 5)

    df.drop(columns=["_nav_for_me"], inplace=True, errors="ignore")
    end_cols = [c for c in df.columns if c not in ("module","possible_module","module_chain","_nav_for_next_str")] \
               + ["module","possible_module","module_chain","_nav_for_next_str"]
    df = df.reindex(columns=end_cols)
    return safe_to_csv(df, out_path, encoding="utf-8")

# ===================== STEP 3: Duplicate call identification =====================
def rows_match_on_spec(prev: pd.Series, cur: pd.Series, evars: List[str], props: List[str]) -> bool:
    for col in evars:
        if not safe_eq(norm(prev.get(col, pd.NA)), norm(cur.get(col, pd.NA))): return False
    for col in props:
        if not safe_eq(norm(prev.get(col, pd.NA)), norm(cur.get(col, pd.NA))): return False
    if not safe_eq(first_nonempty(prev, ["custom_event_name"]), first_nonempty(cur, ["custom_event_name"])): return False
    if not safe_eq(first_nonempty(prev, ["trackaction","trackAction"]), first_nonempty(cur, ["trackaction","trackAction"])): return False
    if not safe_eq(first_nonempty(prev, ["product-string","product","product_string","productString"]),
                   first_nonempty(cur,  ["product-string","product","product_string","productString"])): return False
    if not safe_eq(first_nonempty(prev, ["events"]), first_nonempty(cur, ["events"])): return False
    if not safe_eq(first_nonempty(prev, ["impressionData","impression_data"]),
                   first_nonempty(cur,  ["impressionData","impression_data"])): return False
    return True

def detect_duplicates(input_modules_csv_path: Path, window_secs: int = 30) -> Path:
    out_path = input_modules_csv_path.with_name(f"{input_modules_csv_path.stem}_duplicate_calls.csv")
    row_id_col = "test_case_id"
    df = pd.read_csv(input_modules_csv_path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"]).replace({"": pd.NA})

    # timestamp
    if "__ts" not in df.columns:
        ts_candidates = ["messageReceivedTime","message_received_time","receivedTime","timestamp","event_time","time"]
        found = next((c for c in ts_candidates if c in df.columns), None)
        df["__ts"] = coerce_to_ts(df[found]) if found else pd.NaT
    else:
        df["__ts"] = coerce_to_ts(df["__ts"])

    # row id
    if row_id_col not in df.columns:
        for cand in ["test_case_id","testCaseId","row_id","id"]:
            if cand in df.columns:
                row_id_col = cand
                break
        else:
            row_id_col = "_synthetic_row_id"
            df[row_id_col] = range(1, len(df)+1)

    df = df.sort_values(by=[row_id_col, "__ts"], na_position="last", kind="mergesort").reset_index(drop=True)
    evar_cols = [c for c in df.columns if c.lower().startswith("evar")]
    prop_cols = [c for c in df.columns if c.lower().startswith("prop")]

    df["dup_status"] = "initial"
    df["initial_test_case_id"] = df[row_id_col]

    for i in range(1, len(df)):
        cur = df.loc[i]; prev = df.loc[i-1]
        ta, tb = prev["__ts"], cur["__ts"]
        if pd.isna(ta) or pd.isna(tb): continue
        dt = abs((tb - ta).total_seconds())
        if dt > window_secs: continue
        if rows_match_on_spec(prev, cur, evar_cols, prop_cols):
            df.at[i, "dup_status"] = "duplicate_call"
            df.at[i, "initial_test_case_id"] = df.loc[i-1, "initial_test_case_id"]

    only_dups = df[df["dup_status"] == "duplicate_call"].copy()
    return safe_to_csv(only_dups, out_path, encoding="utf-8")

# ===================== STEP 4: Unique rows from duplicates =====================
IGNORE_COLS_UNIQUE = {
    "__ts","messageReceivedTime","message_received_time","receivedTime",
    "timestamp","event_time","time",
    "initial_test_case_id",
    "module_chain",
    "_nav_for_next_str",
    "test_case_id",
}
_COMPARE_BASE = {
    "custom_event_name",
    "trackaction", "trackAction",
    "events",
    "product-string", "product", "product_string", "productString",
    "impressionData", "impression_data",
    "module", "possible_module",
}

def unique_from_duplicates(input_duplicate_csv_path: Path) -> Path:
    out_path = input_duplicate_csv_path.with_name(f"{input_duplicate_csv_path.stem}_unique_rows.csv")
    df = pd.read_csv(input_duplicate_csv_path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"]).replace({"": pd.NA})

    if df.empty:
        written = safe_to_csv(df, out_path, encoding="utf-8")
        print("\nNo duplicates => Final unique set is empty.\n")
        pretty_print(df)
        print(f"\nSaved: {written}")
        return written

    evar_cols = [c for c in df.columns if c.lower().startswith("evar")]
    prop_cols = [c for c in df.columns if c.lower().startswith("prop")]
    base_cols = [c for c in _COMPARE_BASE if c in df.columns]
    compare_cols = [c for c in (base_cols + evar_cols + prop_cols) if c not in IGNORE_COLS_UNIQUE]

    df_norm = df.copy()
    for c in compare_cols:
        df_norm[c] = df_norm[c].astype("string").str.strip()

    mask = ~df_norm.duplicated(subset=compare_cols, keep="first")
    unique_df = df.loc[mask].copy()

    if "test_case_id" in unique_df.columns:
        try:
            unique_df["test_case_id"] = pd.to_numeric(unique_df["test_case_id"], errors="coerce")
            unique_df = unique_df.sort_values(by=["test_case_id"], kind="mergesort")
        except Exception:
            pass

    written = safe_to_csv(unique_df, out_path, encoding="utf-8")

    print("\n" + "="*100)
    print("FINAL UNIQUE ROWS".center(100))
    print("="*100 + "\n")
    pretty_print(unique_df)
    print(f"\nSaved: {written} | Total rows: {len(unique_df)}")
    print("="*100)
    return written

# ===================== Orchestrator =====================
def run_pipeline(start_path: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Start from .json/.csv/.xlsx, run all steps. Returns:
      (base_csv, modules_csv, dups_csv, unique_csv)
    """
    # Normalize unknown extension files
    start_path = sniff_filetype_and_fix_extension(start_path)

    ext = start_path.suffix.lower()
    if ext == ".json":
        base_csv = parse_jsonish_to_csv(start_path)
    elif ext == ".csv":
        base_csv = start_path
    elif ext == ".xlsx":
        base_csv = excel_to_csv(start_path)
    else:
        raise SystemExit("❌ Input file must have an extension (.json, .csv, .xlsx)")

    modules_csv = classify_modules(base_csv)
    dups_csv    = detect_duplicates(modules_csv, window_secs=30)
    unique_csv  = unique_from_duplicates(dups_csv)
    return base_csv, modules_csv, dups_csv, unique_csv

# ===================== Main =====================
if __name__ == "__main__":
    # Avoid emojis in prompt (Windows cp1252 consoles can choke on them)
    try:
        user_in = input("Enter Google Drive FILE PATH or SHARE LINK (.json/.csv/.xlsx): ").strip()
    except Exception:
        print("Enter Google Drive FILE PATH or SHARE LINK (.json/.csv/.xlsx): ", end="", flush=True)
        user_in = sys.stdin.readline().strip()

    if not user_in:
        raise SystemExit("No input provided.")

    in_path = resolve_input_any(user_in)

    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    # If running in Colab and the file is NOT under Drive, copy into Drive so all outputs land there
    if in_colab() and not str(in_path).startswith("/content/drive"):
        ensure_colab_drive_mounted()
        dst_dir = Path("/content/drive/MyDrive/_pipeline_inputs")
        dst_dir.mkdir(parents=True, exist_ok=True)
        new_path = dst_dir / in_path.name
        shutil.copy2(in_path, new_path)
        print(f"Copied input to Drive: {new_path}")
        in_path = new_path

    base_csv, modules_csv, dups_csv, unique_csv = run_pipeline(in_path)


# In[ ]:




