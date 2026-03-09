"""
src/integrations/sheets_loader.py
===================================
Reads bin records from the Google Sheets analytics hub
("IWMRO_Global_Data" by default, falls back to "IWMRO_Data_Log").

The loader mirrors the column schema written by sheets_logger.py and
returns a cleaned pandas DataFrame ready for the analytics dashboard.

Column mapping
--------------
run_id | bin_id | waste_type | confidence | fill_level |
collection_priority | location_name | latitude | longitude |
truck_id | route_sequence | timestamp

Usage
-----
    from src.integrations.sheets_loader import load_from_sheets
    df = load_from_sheets()           # tries IWMRO_Global_Data first
    df = load_from_sheets("IWMRO_Data_Log")   # explicit name
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SHEETS_CREDS_FILE

logger = logging.getLogger(__name__)

_SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

# Ordered list of spreadsheet names to try (most preferred first)
_SPREADSHEET_CANDIDATES = [
    "IWMRO_Global_Data",
    "IWMRO_Data_Log",
]

_NUMERIC_COLS = ["fill_level", "confidence", "latitude", "longitude", "route_sequence"]


def _get_client():
    """Return an authorised gspread client or raise on failure."""
    from oauth2client.service_account import ServiceAccountCredentials
    import gspread

    if not SHEETS_CREDS_FILE.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {SHEETS_CREDS_FILE}. "
            "Follow the setup steps in sheets_logger.py."
        )
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        str(SHEETS_CREDS_FILE), _SCOPE
    )
    return gspread.authorize(creds)


def load_from_sheets(
    spreadsheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download all bin records from the Google Sheets hub.

    Parameters
    ----------
    spreadsheet_name : Override the default lookup order.  If None, tries
                       IWMRO_Global_Data then IWMRO_Data_Log.

    Returns
    -------
    pd.DataFrame — cleaned bin records (empty DataFrame on any failure).
    """
    if not SHEETS_CREDS_FILE.exists():
        logger.info("[SheetsLoader] creds.json absent — skipping Sheets load.")
        return pd.DataFrame()

    try:
        client = _get_client()
    except ImportError as exc:
        logger.warning("[SheetsLoader] gspread/oauth2client not installed: %s", exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("[SheetsLoader] Auth failed: %s", exc)
        return pd.DataFrame()

    candidates = [spreadsheet_name] if spreadsheet_name else _SPREADSHEET_CANDIDATES

    sheet = None
    used_name = None
    for name in candidates:
        try:
            sheet = client.open(name).sheet1
            used_name = name
            break
        except Exception:
            logger.debug("[SheetsLoader] Spreadsheet '%s' not found — trying next.", name)

    if sheet is None:
        logger.warning(
            "[SheetsLoader] None of the candidate spreadsheets were accessible: %s",
            candidates,
        )
        return pd.DataFrame()

    try:
        records = sheet.get_all_records()   # row 1 = headers; gspread returns list[dict]
    except Exception as exc:
        logger.warning("[SheetsLoader] Failed to read records from '%s': %s", used_name, exc)
        return pd.DataFrame()

    if not records:
        logger.info("[SheetsLoader] Spreadsheet '%s' is empty.", used_name)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Coerce numeric columns ────────────────────────────────────────────────
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows without valid coordinates ───────────────────────────────────
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])

    # ── Normalise collection_priority casing ─────────────────────────────────
    if "collection_priority" in df.columns:
        df["collection_priority"] = df["collection_priority"].str.lower().fillna("normal")

    logger.info(
        "[SheetsLoader] Loaded %d records from '%s'.",
        len(df), used_name,
    )
    return df.reset_index(drop=True)
