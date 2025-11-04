#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_gdrive_full_unified.py

Unified pipeline with Google Drive I/O (Jupyter/Colab friendly)
Steps:
  1) Parse semi-JSON/text → CSV with Excel-safe IST TIMESTAMP
  2) Classify modules (module right after custom_event_name; tails at the end)
  3) Detect adjacent duplicates (row i vs i-1 within N seconds)
  4) Consolidate unique-from-duplicates

- Accepts Google Drive SHARE LINK or Drive/local PATH (.json/.csv/.xlsx)
- Uploads outputs back to the SAME Drive folder as the shared input
- Prints Adjacent Duplicates (preview) and FINAL UNIQUE DUPLICATE ROWS in a BOX
"""

from __future__ import annotations

import os
import re
import sys
import io
import math
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Set, Dict
from collections import Counter

import pandas as pd

# --------------------------------------------------------------------------------------
# Pretty print (tabulate if available) — BOXED output
# --------------------------------------------------------------------------------------
def pretty_print(df: pd.DataFrame, limit: int = 200, title: str | None = None):
    if title:
        bar = "=" * max(80, len(title))
        print("\n" + bar)
        print(title)
        print(bar)
    try:
        from tabulate import tabulate  # type: ignore
        print(tabulate(df.head(limit), headers="keys", tablefmt="fancy_grid", showindex=False))
    except Exception:
        s = df.head(limit).to_string(index=False)
        line = "=" * max(80, len(s.splitlines()[0]) if s else 80)
        print(line); print(s); print(line)

# --------------------------------------------------------------------------------------
# Environment detection + Drive helpers
# --------------------------------------------------------------------------------------
def pip_install(packages: List[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])

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
    """
    If in Colab and relative path is provided, assume /content/drive/MyDrive base.
    Otherwise, treat as-is.
    """
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
    def __init__(self, path: Path):
        self.path = path
        self._old = None
    def __enter__(self):
        self._old = Path.cwd()
        os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)
    def __exit__(self, exc_type, exc, tb):
        if self._old:
            os.chdir(self._old)

def download_from_drive_url(url: str) -> Tuple[Path, str]:
    """
    Download a Drive share link via gdown. Returns (local_file_path, file_id).
    """
    m = re.search(r"/d/([^/]+)/", url) or re.search(r"[?&]id=([^&]+)", url)
    if not m:
        raise SystemExit("ERROR: Could not extract file id from the Drive link.")
    file_id = m.group(1)

    if not have_gdown():
        print("Installing gdown ...")
        pip_install(["gdown"])

    import gdown

    # Choose a download directory
    if in_colab():
        ensure_colab_drive_mounted()
        dl_dir = Path("/content/drive/MyDrive/_pipeline_inputs")
    else:
        dl_dir = Path.home() / "Downloads" / "_pipeline_inputs"
    dl_dir.mkdir(parents=True, exist_ok=True)

    with _ChDir(dl_dir):
        out = gdown.download(id=file_id, output=None, quiet=False, fuzzy=True)
        if not out:
            raise SystemExit("ERROR: Download failed. Make sure link is accessible.")
        p = (dl_dir / out).resolve()
        if not p.exists():
            cand = dl_dir / "downloaded_file"
            if cand.exists():
                p = cand
            else:
                raise SystemExit("ERROR: Download reported success but file not found.")
    print(f"✅ Download complete! Saved to: {p}")
    return p, file_id

def resolve_input_any(user_text: str) -> Tuple[Path, Optional[str]]:
    """
    If Drive share link → download via gdown and return (local_path, file_id).
    Else → return (resolved_path, None).
    """
    s = user_text.strip().strip('"').strip("'")
    if "drive.google.com" in s:
        p, fid = download_from_drive_url(s)
        return p, fid
    return resolve_gdrive_path(s), None

# --------------------------------------------------------------------------------------
# Upload outputs back to SAME Drive folder as the shared input
# --------------------------------------------------------------------------------------
def upload_outputs_back_same_folder_colab(file_id: str, local_paths: List[Path]) -> List[str]:
    """
    Colab: use google-api-python-client with built-in auth.
    """
    try:
        from google.colab import auth as colab_auth  # type: ignore
        colab_auth.authenticate_user()
    except Exception as e:
        raise RuntimeError(f"Colab auth failed: {e}")

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except Exception:
        pip_install(["google-api-python-client"])
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

    drive_service = build("drive", "v3")
    meta = drive_service.files().get(fileId=file_id, fields="parents").execute()
    parents = meta.get("parents", [])
    parent_id = parents[0] if parents else None
    if not parent_id:
        print("⚠️ Couldn't resolve parent folder of shared file; uploading to your Drive root.")

    uploaded_ids = []
    for p in local_paths:
        media = MediaFileUpload(str(p), resumable=True)
        body = {"name": p.name}
        if parent_id:
            body["parents"] = [parent_id]
        created = drive_service.files().create(body=body, media_body=media, fields="id").execute()
        fid = created["id"]
        uploaded_ids.append(fid)
        print(f"☁️ Uploaded to Drive: {p.name} (id={fid})")
    return uploaded_ids

def upload_outputs_back_same_folder_local(file_id: str, local_paths: List[Path]) -> List[str]:
    """
    Local: use pydrive2 (first run may open a browser for OAuth).
    """
    try:
        import pydrive2  # noqa
    except Exception:
        pip_install(["pydrive2"])
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive

    ga = GoogleAuth()
    ga.LoadCredentialsFile("credentials.json")
    if ga.credentials is None:
        ga.LocalWebserverAuth()
    elif ga.access_token_expired:
        ga.Refresh()
    else:
        ga.Authorize()
    ga.SaveCredentialsFile("credentials.json")

    drive = GoogleDrive(ga)
    f = drive.CreateFile({'id': file_id})
    f.FetchMetadata(fields='parents')
    parents = f.get('parents') or []
    parent_id = parents[0].get('id') if parents else None

    uploaded = []
    for p in local_paths:
        meta = {'title': p.name}
        if parent_id:
            meta['parents'] = [{'id': parent_id}]
        gfile = drive.CreateFile(meta)
        gfile.SetContentFile(str(p))
        gfile.Upload()
        uploaded.append(gfile['id'])
        print(f"☁️ Uploaded to Drive: {p.name} (id={gfile['id']})")
    return uploaded

# --------------------------------------------------------------------------------------
# File-type detection & CSV writer
# --------------------------------------------------------------------------------------
def sniff_filetype_and_fix_extension(path: Path) -> Path:
    """
    If file has no extension, try to infer and rename.
    """
    if path.suffix:
        return path
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:2] == b"PK":
        newp = path.with_suffix(".xlsx")
        path.rename(newp)
        print(f"ℹ️ Inferred XLSX and renamed to: {newp.name}")
        return newp
    text = head.decode("utf-8", errors="ignore").lstrip()
    if text.startswith("{") or text.startswith("[" ):
        newp = path.with_suffix(".json")
        path.rename(newp)
        print(f"ℹ️ Inferred JSON and renamed to: {newp.name}")
        return newp
    newp = path.with_suffix(".csv")
    path.rename(newp)
    print(f"ℹ️ Inferred CSV and renamed to: {newp.name}")
    return newp

def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """
    Write CSV with fallbacks for locked files.
    """
    path = Path(path)
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
    raise PermissionError(f"Could not write CSV (locked?). Last error: {last_err}")

# --------------------------------------------------------------------------------------
# Stage 1 — Parse semi-JSON/text → CSV with IST TIMESTAMP (Excel-safe)
# --------------------------------------------------------------------------------------
TESTCASE_DELIMS = ["]}}]}", "]}}],", "]}}]}", "]}}]},", "]}},"]

RE_EVAR = re.compile(r'"(eVar\d+)"\s*:\s*"([^"]*)"')
RE_PROP = re.compile(r'"(prop\d+)"\s*:\s*"([^"]*)"')
RE_EVENTS = re.compile(r'"events"\s*:\s*"([^"]*)"')
RE_PRODUCT_STRING = re.compile(r'"product-string"\s*:\s*"([^"]*)"')
RE_TRACK_ACTION = re.compile(r'"trackAction"\s*:\s*"([^"]*)"|"trackaction"\s*:\s*"([^"]*)"')
RE_MSG_TIME = re.compile(r'"messageReceivedTime"\s*:\s*\{\s*"date"\s*:\s*"([^"]*)"')
RE_IMPRESSION_DATA = re.compile(r'"impressionData"\s*:\s*"([^"]*)"')
RE_CUSTOM_EVENT = re.compile(r'"custom_event_name"\s*:\s*"([^"]*)"')
RE_TS_MS = re.compile(r'"timestamp"\s*:\s*(\d{10,}(?:\.\d+)?)')

def parse_raw_to_csv(input_path: Path) -> Path:
    """
    Produces:
      - shifted 'timestamp' (up by one row)
      - Excel-safe 'TIMESTAMP' in IST with ms (UTC -> IST, ceil ms + 1)
    """
    input_path = Path(input_path)
    out_path = input_path.with_name(f"{input_path.stem}_parsed.csv")

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    delim = next((d for d in TESTCASE_DELIMS if d in text), TESTCASE_DELIMS[0])
    chunks = text.split(delim)

    rows = []
    case = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        case += 1
        row = {"test_case_id": case}

        for k, v in RE_EVAR.findall(chunk):
            row[k] = v
        for k, v in RE_PROP.findall(chunk):
            row[k] = v

        for pat, name in [
            (RE_EVENTS, "events"),
            (RE_PRODUCT_STRING, "product-string"),
            (RE_IMPRESSION_DATA, "impressionData"),
            (RE_CUSTOM_EVENT, "custom_event_name"),
            (RE_MSG_TIME, "messageReceivedTime"),
        ]:
            m = pat.search(chunk)
            if m:
                row[name] = m.group(1)

        m = RE_TRACK_ACTION.search(chunk)
        if m:
            row["trackaction"] = m.group(1) or m.group(2)

        m = RE_TS_MS.search(chunk)
        if m:
            row["timestamp"] = m.group(1)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Shift and compute IST TIMESTAMP (rounded up ms + 1)
    if "timestamp" in df.columns and not df.empty:
        df["timestamp"] = df["timestamp"].shift(-1)
        df.loc[df.index[-1], "timestamp"] = pd.NA

        ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
        ts_num = ts_num.apply(lambda x: math.ceil(x) + 1 if pd.notna(x) else pd.NA)
        ts_utc = pd.to_datetime(ts_num, unit="ms", errors="coerce", utc=True)
        ts_ist = ts_utc.dt.tz_convert("Asia/Kolkata")
        df["TIMESTAMP"] = "'" + ts_ist.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    else:
        df["TIMESTAMP"] = pd.NA

    # Column order
    def sort_cols(cols: List[str]) -> List[str]:
        lead = [
            "test_case_id",
            "TIMESTAMP",
            "messageReceivedTime",
            "trackaction",
            "custom_event_name",
            "events",
            "product-string",
            "impressionData",
            "timestamp",
        ]
        ev = sorted([c for c in cols if re.match(r"eVar\d+$", c, re.I)])
        pr = sorted([c for c in cols if re.match(r"prop\d+$", c, re.I)])
        rest = [c for c in cols if c not in set(lead + ev + pr)]
        return [c for c in lead if c in cols] + ev + pr + rest

    if not df.empty:
        df = df.reindex(columns=sort_cols(list(df.columns)))

    written = safe_to_csv(df, out_path, encoding="utf-8")
    print(f"✅ Parsed saved → {written}")
    return written

# --------------------------------------------------------------------------------------
# Stage 2 — Module classification (ordering fixed per your requirement)
# --------------------------------------------------------------------------------------
MODULE_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
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

def _norm_text(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower()

def _mods_in_text(text: str) -> Set[str]:
    t = _norm_text(text)
    if not t:
        return set()
    hits = set()
    for mod, kw in MODULE_KEYWORDS.items():
        if any(w in t for w in kw["custom"] + kw["action"]):
            hits.add(mod)
    return hits

def _text_modules(row: pd.Series) -> List[str]:
    ce = row.get("custom_event_name", "")
    ta = row.get("trackaction", row.get("trackAction", ""))
    if "tab bar bottom nav" in _norm_text(ta):
        return ["Navigation"]
    return sorted(_mods_in_text(ce) | _mods_in_text(ta))

def _first_priority_modules(row: pd.Series) -> List[str]:
    for col in PRIORITY_COLS:
        if col in row:
            mods = sorted(_mods_in_text(row[col]))
            if mods:
                return mods
    return []

def _e43_45_single(row: pd.Series) -> List[str]:
    mods = _mods_in_text(row.get("eVar43", "")) | _mods_in_text(row.get("eVar45", ""))
    return list(mods) if len(mods) == 1 else []

def _insert_after(cols: list[str], anchor: str, item: str) -> list[str]:
    cols = [c for c in cols if c != item]
    if anchor in cols:
        i = cols.index(anchor) + 1
        return cols[:i] + [item] + cols[i:]
    return cols + [item]

def classify_modules(parsed_csv: Path) -> Path:
    parsed_csv = Path(parsed_csv)
    out_path = parsed_csv.with_name(f"{parsed_csv.stem}_modules.csv")

    df = pd.read_csv(parsed_csv, dtype=str, keep_default_na=False).replace({"": pd.NA})
    original_order = list(df.columns)  # preserve

    # --- initial module inference ---
    initial_modules: List[str] = []
    for _, r in df.iterrows():
        m = _text_modules(r)
        if not m:
            m = _first_priority_modules(r)
        initial_modules.append(", ".join(m) if m else "")
    df["module"] = initial_modules

    # --- overrides ---
    for i, r in df.iterrows():
        p43 = _norm_text(r.get("prop43", "")); p45 = _norm_text(r.get("prop45", ""))
        if "account created" in p43 or "|account created" in p45:
            df.at[i, "module"] = "My account"

    for i, r in df.iterrows():
        cur = df.at[i, "module"] or ""
        if "," in cur or cur == "":
            s = _e43_45_single(r)
            if s:
                df.at[i, "module"] = s[0]

    for i, r in df.iterrows():
        if df.at[i, "module"] in ("", "Unknown"):
            e216 = _norm_text(r.get("eVar216", ""))
            if "sb_zone" in e216:
                df.at[i, "module"] = "Bag"
            elif "hp_zone" in e216:
                df.at[i, "module"] = "Home page"

    df["module"] = df["module"].replace({"": "Unknown"})

    # --- look-ahead/back fill aids ---
    nav_for_next = df.apply(
        lambda r: ", ".join(sorted(
            (_mods_in_text(r.get("eVar29", "")) | _mods_in_text(r.get("eVar30", "")))
        )) or "",
        axis=1,
    )
    df["_nav_for_next_str"] = nav_for_next
    df["_nav_for_me"] = df["_nav_for_next_str"].shift(1)
    mask_fill = (df["module"] == "Unknown") & df["_nav_for_me"].fillna("").ne("")
    df.loc[mask_fill, "module"] = df.loc[mask_fill, "_nav_for_me"]

    # --- possible_module & module_chain for Unknowns ---
    def window_modules(idx: int, k: int = 5) -> List[str]:
        lo, hi = max(0, idx - k), min(len(df), idx + k + 1)
        return [df.at[j, "module"] for j in range(lo, hi) if j != idx]

    def most_freq(mod_list: List[str]) -> str:
        clean = [m for m in mod_list if m and m != "Unknown"]
        if not clean:
            return ""
        c = Counter(clean)
        m = max(c.values())
        return ", ".join(sorted([k for k, v in c.items() if v == m]))

    def chain_view(idx: int, k: int = 5) -> str:
        n = len(df)
        next_idxs = list(range(idx + 1, min(n, idx + 1 + k)))
        next_part = [str(df.at[j, "module"]) for j in reversed(next_idxs)]
        prev_part = [str(df.at[j, "module"]) for j in range(idx - 1, max(-1, idx - 1 - k), -1)]
        return " -> ".join(next_part + ["<<NO MODULE>>"] + prev_part)

    df["possible_module"] = ""
    df["module_chain"] = ""
    unknown_mask = df["module"].eq("Unknown")
    for i in df.index[unknown_mask]:
        prev_mod = df.at[i-1, "module"] if i-1 >= 0 else ""
        next_mod = df.at[i+1, "module"] if i+1 < len(df) else ""
        if prev_mod and next_mod and prev_mod != "Unknown" and next_mod != "Unknown" and prev_mod == next_mod:
            df.at[i, "possible_module"] = prev_mod
        else:
            df.at[i, "possible_module"] = most_freq(window_modules(i))
        df.at[i, "module_chain"] = chain_view(i)

    # --- final column ordering rules ---
    for c in ["module", "possible_module", "module_chain", "_nav_for_next_str", "custom_event_name"]:
        if c not in df.columns:
            df[c] = pd.NA

    final_cols = original_order.copy()

    # place 'module' immediately AFTER 'custom_event_name'
    final_cols = _insert_after(final_cols, "custom_event_name", "module")

    # force tail columns at the END (and in this exact order)
    for tail in ["possible_module", "module_chain", "_nav_for_next_str"]:
        if tail in final_cols:
            final_cols.remove(tail)
    final_cols = [c for c in final_cols if c in df.columns]
    final_cols += ["possible_module", "module_chain", "_nav_for_next_str"]

    # append any new/unseen columns just before the enforced tail trio
    seen = set(final_cols)
    for c in df.columns:
        if c not in seen:
            final_cols.insert(-3, c)

    df.drop(columns=["_nav_for_me"], inplace=True, errors="ignore")
    df = df.reindex(columns=final_cols)

    written = safe_to_csv(df, out_path, encoding="utf-8")
    print(f"✅ Module classified (ordering fixed) → {written}")
    return written

# --------------------------------------------------------------------------------------
# Stage 3 — Adjacent duplicate detection (compare i vs i-1 within window seconds)
# --------------------------------------------------------------------------------------
def _norm_val(v):
    return pd.NA if pd.isna(v) or str(v).strip() == "" else str(v).strip()

def _safe_eq(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b

def _coerce_ts(series: pd.Series) -> pd.Series:
    # try epoch ms first
    num = pd.to_numeric(series, errors="coerce")
    ts = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
    if ts.notna().any():
        return ts
    # fallback
    return pd.to_datetime(series, errors="coerce", utc=True)

def detect_adjacent_duplicates(path: Path, window: int = 5) -> Path:
    path = Path(path)
    out = path.with_name(f"{path.stem}_adjacent_dups_{window}s.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False).replace({"": pd.NA})

    if "timestamp" not in df.columns:
        raise SystemExit("❌ Required column 'timestamp' not found.")
    df["__ts"] = _coerce_ts(df["timestamp"])

    ev = [c for c in df.columns if c.lower().startswith("evar")]
    pr = [c for c in df.columns if c.lower().startswith("prop")]
    fixed = [c for c in ["trackaction", "custom_event_name", "module", "events", "product-string", "impressionData"] if c in df.columns]
    cmp_cols = fixed + ev + pr

    dup = [False] * len(df)
    for i in range(1, len(df)):
        a, b = df.loc[i - 1], df.loc[i]
        ta, tb = a["__ts"], b["__ts"]
        if pd.isna(ta) or pd.isna(tb):
            continue
        if abs((tb - ta).total_seconds()) > window:
            continue
        if all(_safe_eq(_norm_val(a.get(c, pd.NA)), _norm_val(b.get(c, pd.NA))) for c in cmp_cols):
            dup[i] = True

    dup_df = df.loc[dup].drop(columns=["__ts"]).copy()
    written = safe_to_csv(dup_df, out, encoding="utf-8")

    # BOXED preview for Adjacent Duplicates
    pretty_print(dup_df, limit=200, title=f"ADJACENT DUPLICATES (≤{window}s) — preview (rows: {len(dup_df)})")
    print(f"💾 Saved Adjacent Duplicates → {written}")
    return written

# --------------------------------------------------------------------------------------
# Stage 4 — Unique-from-duplicates consolidation (BOXED print)
# --------------------------------------------------------------------------------------
IGNORE_COLS_UNIQUE = {
    "TIMESTAMP","messageReceivedTime","message_received_time","receivedTime",
    "timestamp","event_time","time",
    "initial_test_case_id",
    "module_chain",
    "_nav_for_next_str",
    "test_case_id",
}
COMPARE_BASE = {
    "custom_event_name",
    "trackaction", "trackAction",
    "events",
    "product-string", "product", "product_string", "productString",
    "impressionData", "impression_data",
    "module", "possible_module",
}

def unique_from_duplicates(dups_csv: Path) -> Path:
    dups_csv = Path(dups_csv)
    out = dups_csv.with_name(f"{dups_csv.stem}_unique_rows.csv")

    df = pd.read_csv(
        dups_csv,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "NaN"]
    ).replace({"": pd.NA})

    if df.empty:
        written = safe_to_csv(df.head(0), out, encoding="utf-8")
        pretty_print(df, title="FINAL UNIQUE DUPLICATE ROWS — (none)")
        print(f"💾 Saved empty set → {written}")
        return written

    evar_cols = [c for c in df.columns if c.lower().startswith("evar")]
    prop_cols = [c for c in df.columns if c.lower().startswith("prop")]
    base_cols = [c for c in COMPARE_BASE if c in df.columns]
    compare_cols = [c for c in (base_cols + evar_cols + prop_cols) if c not in IGNORE_COLS_UNIQUE]
    if not compare_cols:
        compare_cols = [c for c in df.columns if c not in IGNORE_COLS_UNIQUE]

    dfn = df.copy()
    for c in compare_cols:
        dfn[c] = dfn[c].astype("string").str.strip().str.lower()

    dupmask = dfn.duplicated(subset=compare_cols, keep=False)
    groups = df[dupmask].copy()

    if groups.empty:
        written = safe_to_csv(df.head(0), out, encoding="utf-8")
        pretty_print(groups, title="FINAL UNIQUE DUPLICATE ROWS — (no groups detected)")
        print(f"💾 Saved empty set → {written}")
        return written

    uniq = groups.drop_duplicates(subset=compare_cols, keep="first").copy()
    if "test_case_id" in uniq.columns:
        try:
            uniq["test_case_id"] = pd.to_numeric(uniq["test_case_id"], errors="coerce")
            uniq = uniq.sort_values(by=["test_case_id"], kind="mergesort")
        except Exception:
            pass

    written = safe_to_csv(uniq, out, encoding="utf-8")

    # BOXED final unique rows
    pretty_print(uniq, limit=200, title=f"FINAL UNIQUE DUPLICATE ROWS — total: {len(uniq)}")
    print(f"💾 Saved Unique-from-duplicates → {written}")
    return written

# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def run_pipeline(start_path: Path, dup_window_secs: int = 5) -> Tuple[Path, Path, Path, Path]:
    """
    Returns (parsed_csv, modules_csv, adjacent_dups_csv, unique_from_dups_csv)
    """
    start_path = sniff_filetype_and_fix_extension(start_path)
    ext = start_path.suffix.lower()
    if ext == ".json":
        parsed_csv = parse_raw_to_csv(start_path)
    elif ext == ".csv":
        # keep naming consistent; if raw CSV, copy to *_parsed.csv
        parsed_csv = start_path.with_name(f"{start_path.stem}_parsed.csv") \
            if "parsed" not in start_path.stem else start_path
        if parsed_csv != start_path and not parsed_csv.exists():
            shutil.copy2(start_path, parsed_csv)
    elif ext == ".xlsx":
        df_x = pd.read_excel(start_path, sheet_name=0, dtype=str).replace({"": pd.NA})
        parsed_csv = safe_to_csv(df_x, start_path.with_suffix(".csv"), encoding="utf-8")
    else:
        raise SystemExit("❌ Input must be .json/.csv/.xlsx (or no-ext that can be sniffed).")

    modules_csv = classify_modules(parsed_csv)
    dups_csv    = detect_adjacent_duplicates(modules_csv, window=dup_window_secs)
    unique_csv  = unique_from_duplicates(dups_csv)
    return parsed_csv, modules_csv, dups_csv, unique_csv

# --------------------------------------------------------------------------------------
# Main (prompt-based; notebook-friendly)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Ignore any Jupyter kernel args
    sys.argv = [sys.argv[0]]

    # Prompt for input path or Drive share link
    try:
        user_in = input("Enter Google Drive FILE PATH or SHARE LINK (.json/.csv/.xlsx): ").strip()
    except Exception:
        print("Enter Google Drive FILE PATH or SHARE LINK (.json/.csv/.xlsx): ", end="", flush=True)
        user_in = sys.stdin.readline().strip()

    if not user_in:
        raise SystemExit("No input provided.")

    # Optional: dup window seconds
    try:
        window_txt = input("Enter adjacent-duplicate window seconds (default 5): ").strip()
        dup_window_secs = int(window_txt) if window_txt else 5
    except Exception:
        dup_window_secs = 5

    # Resolve to a real file and (if link) capture Drive file id
    in_path, drive_file_id = resolve_input_any(user_in)
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    # In Colab: ensure outputs are saved into Drive (copy working input into Drive if needed)
    if in_colab() and not str(in_path).startswith("/content/drive"):
        ensure_colab_drive_mounted()
        dst_dir = Path("/content/drive/MyDrive/_pipeline_inputs")
        dst_dir.mkdir(parents=True, exist_ok=True)
        new_path = dst_dir / in_path.name
        shutil.copy2(in_path, new_path)
        print(f"Copied working input into Drive so outputs land in Drive: {new_path}")
        in_path = new_path

    # Run pipeline
    parsed_csv, modules_csv, dups_csv, unique_csv = run_pipeline(in_path, dup_window_secs=dup_window_secs)

    # Upload outputs back to the SAME shared folder if a share link was used
    if drive_file_id:
        try:
            if in_colab():
                uploaded_ids = upload_outputs_back_same_folder_colab(
                    drive_file_id, [parsed_csv, modules_csv, dups_csv, unique_csv]
                )
            else:
                uploaded_ids = upload_outputs_back_same_folder_local(
                    drive_file_id, [parsed_csv, modules_csv, dups_csv, unique_csv]
                )
            print("\nDrive uploads complete (same folder as the shared input):")
            for p, fid in zip([parsed_csv, modules_csv, dups_csv, unique_csv], uploaded_ids):
                print(f"  - {p.name}  (file id: {fid})")
        except Exception as e:
            print(f"\n⚠️ Upload back to the shared file's folder failed: {e}")
            print("Outputs are still saved next to your working input shown above.")
