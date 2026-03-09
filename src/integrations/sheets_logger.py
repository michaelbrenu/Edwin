"""
src/integrations/sheets_logger.py
===================================
Pushes pipeline results to a Google Sheets spreadsheet so every
run is automatically visible in the "automation hub" sheet.

Sheet layout (IWMRO_Data_Log)
------------------------------
Row 1  — headers (written once, never overwritten)
Row 2+ — one row per bin record, appended on every pipeline run

Columns
-------
run_id | bin_id | waste_type | confidence | fill_level |
collection_priority | location_name | latitude | longitude |
truck_id | route_sequence | timestamp

Credentials
-----------
Download a Google Cloud service-account JSON key and save it as
`creds.json` in the project root (same folder as app.py).

  1. Google Cloud Console → IAM → Service Accounts → Create key (JSON)
  2. Share the target spreadsheet with the service-account e-mail
     (Editor role is required to append rows)
  3. Save the JSON file as  <project_root>/creds.json

If creds.json is absent the logger silently skips — the rest of
the pipeline is not affected.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SHEETS_SPREADSHEET, SHEETS_CREDS_FILE

logger = logging.getLogger(__name__)

# Column headers written to row 1 of the sheet (order matters)
_SHEET_HEADERS = [
    "run_id", "bin_id", "waste_type", "confidence",
    "fill_level", "collection_priority",
    "location_name", "latitude", "longitude",
    "truck_id", "route_sequence", "timestamp",
]

_SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


def _get_client():
    """Return an authorised gspread client or raise ImportError / FileNotFoundError."""
    from oauth2client.service_account import ServiceAccountCredentials
    import gspread

    if not SHEETS_CREDS_FILE.exists():
        raise FileNotFoundError(
            f"Google Sheets credentials not found at {SHEETS_CREDS_FILE}. "
            "See the module docstring for setup instructions."
        )

    creds  = ServiceAccountCredentials.from_json_keyfile_name(
        str(SHEETS_CREDS_FILE), _SCOPE
    )
    return gspread.authorize(creds)


def _ensure_headers(sheet) -> None:
    """Write column headers to row 1 if the sheet is empty."""
    first_row = sheet.row_values(1)
    if not first_row or first_row[0] != "run_id":
        sheet.insert_row(_SHEET_HEADERS, index=1)
        logger.info("[Sheets] Headers written to row 1.")


def log_bins_to_sheet(
    bins: list[dict],
    run_id: Optional[str] = None,
) -> int:
    """
    Append one row per bin to the Google Sheet.

    Parameters
    ----------
    bins   : annotated bin records from the pipeline
    run_id : identifier for this pipeline run (defaults to current ISO timestamp)

    Returns
    -------
    int — number of rows appended (0 if skipped due to missing credentials)
    """
    if not SHEETS_CREDS_FILE.exists():
        logger.info(
            "[Sheets] creds.json not found — skipping Google Sheets sync. "
            "See src/integrations/sheets_logger.py for setup instructions."
        )
        return 0

    try:
        client = _get_client()
    except ImportError as exc:
        logger.warning("[Sheets] gspread / oauth2client not installed: %s", exc)
        return 0
    except Exception as exc:
        logger.warning("[Sheets] Could not authorise Google Sheets client: %s", exc)
        return 0

    try:
        sheet = client.open(SHEETS_SPREADSHEET).sheet1
    except Exception as exc:
        logger.warning("[Sheets] Could not open spreadsheet '%s': %s",
                       SHEETS_SPREADSHEET, exc)
        return 0

    _ensure_headers(sheet)

    rid = run_id or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Build rows in header order
    rows = [
        [
            rid,
            b.get("bin_id", ""),
            b.get("waste_type", ""),
            b.get("confidence", ""),
            b.get("fill_level", ""),
            b.get("collection_priority", ""),
            b.get("location_name", ""),
            b.get("latitude", ""),
            b.get("longitude", ""),
            b.get("truck_id", ""),
            b.get("route_sequence", ""),
            b.get("timestamp", ""),
        ]
        for b in bins
    ]

    # gspread append_rows is a single API call — much faster than row-by-row
    sheet.append_rows(rows, value_input_option="USER_ENTERED")
    logger.info("[Sheets] Appended %d rows to '%s'.", len(rows), SHEETS_SPREADSHEET)
    return len(rows)
